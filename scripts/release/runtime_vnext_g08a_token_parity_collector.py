#!/usr/bin/env python3
"""Collect exact G08A Ferrum/llama.cpp prompt and greedy-token parity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import runtime_vnext_g08a_numerics as numerics


PASS_PREFIX = "FERRUM RUNTIME VNEXT G08A TOKEN PARITY COLLECTOR PASS"
SELFTEST_PASS = "FERRUM RUNTIME VNEXT G08A TOKEN PARITY COLLECTOR SELFTEST PASS"
MODEL_ALIAS = "qwen3.5:4b-q4_k_m"
REFERENCE_TEMPLATE_KWARGS = '{"enable_thinking":false}'
CONTEXT_SIZE = 1024
THREAD_LIMIT = 4
SERVER_START_TIMEOUT_SECONDS = 120
REQUEST_TIMEOUT_SECONDS = 300
PROCESS_STOP_TIMEOUT_SECONDS = 15
FERRUM_ENV_LIMITS = {
    "RAYON_NUM_THREADS": str(THREAD_LIMIT),
    "TOKIO_WORKER_THREADS": "2",
    "OMP_NUM_THREADS": str(THREAD_LIMIT),
    "MKL_NUM_THREADS": str(THREAD_LIMIT),
    "VECLIB_MAXIMUM_THREADS": str(THREAD_LIMIT),
}


class CollectorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectorError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_json_bytes(value))


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CollectorError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def git_value(source: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(source), "-c", "core.preloadindex=false", "-c", "index.threads=1", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(process.returncode == 0, process.stderr.strip() or f"git {' '.join(args)} failed")
    return process.stdout.strip()


def clean_git_identity(source: Path, label: str) -> dict[str, Any]:
    source = source.resolve()
    sha = git_value(source, "rev-parse", "HEAD")
    status = git_value(source, "status", "--short")
    require(numerics.GIT_SHA_RE.fullmatch(sha) is not None, f"{label} git SHA is invalid")
    require(not status, f"{label} source must be clean; dirty paths:\n{status}")
    return {"path": str(source), "git_sha": sha, "dirty": False}


def binary_identity(path: Path, label: str) -> dict[str, str]:
    path = path.expanduser().resolve()
    require(path.is_file() and os.access(path, os.X_OK), f"{label} is not executable: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def locked_model_path(path: Path, revision: str, expected_sha256: str) -> Path:
    snapshot_path = Path(os.path.abspath(path.expanduser()))
    resolved = snapshot_path.resolve()
    require(resolved.is_file(), f"GGUF model is unavailable: {resolved}")
    require(snapshot_path.parent.name == revision, "GGUF snapshot revision differs from the M1 lock")
    require(sha256_file(resolved) == expected_sha256, "GGUF model SHA256 differs from the M1 lock")
    return resolved


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    data = None if payload is None else canonical_json_bytes(payload)
    request = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise CollectorError(f"HTTP {path} failed: {error}") from error
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise CollectorError(f"HTTP {path} returned invalid JSON: {body[:500]!r}") from error
    require(isinstance(value, dict), f"HTTP {path} response must be an object")
    return value


def reference_server_contract(binary: str) -> list[str]:
    return [
        binary,
        "--model",
        "MODEL",
        "--host",
        "127.0.0.1",
        "--port",
        "PORT",
        "--ctx-size",
        str(CONTEXT_SIZE),
        "--parallel",
        "1",
        "--threads",
        str(THREAD_LIMIT),
        "--threads-batch",
        str(THREAD_LIMIT),
        "--n-gpu-layers",
        "99",
        "--jinja",
        "--chat-template-kwargs",
        REFERENCE_TEMPLATE_KWARGS,
        "--reasoning",
        "off",
        "--no-warmup",
        "--cache-ram",
        "0",
    ]


def reference_server_argv(binary: Path, model: Path, port: int) -> list[str]:
    return [
        str(model) if token == "MODEL" else str(port) if token == "PORT" else token
        for token in reference_server_contract(str(binary))
    ]


def ferrum_command_contract(binary: str) -> list[str]:
    return [
        binary,
        "run",
        MODEL_ALIAS,
        "--backend",
        "metal",
        "--prompt",
        "PROMPT",
        "--request-dump-dir",
        "REQUEST_DUMP_DIR",
        "--disable-thinking",
        "--max-tokens",
        str(numerics.TOKEN_COUNT),
        "--temperature",
        "0",
        "--seed",
        "9271",
        "--top-k",
        "0",
        "--top-p",
        "1",
        "--min-p",
        "0",
        "--presence-penalty",
        "0",
        "--repeat-penalty",
        "1",
        "--max-model-len",
        str(CONTEXT_SIZE),
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        str(CONTEXT_SIZE),
        "--kv-capacity",
        "8192",
        "--output-format",
        "jsonl",
    ]


def ferrum_argv(binary: Path, prompt: str, dump_dir: Path) -> list[str]:
    return [
        prompt if token == "PROMPT" else str(dump_dir) if token == "REQUEST_DUMP_DIR" else token
        for token in ferrum_command_contract(str(binary))
    ]


def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    for sig, timeout in (
        (signal.SIGINT, PROCESS_STOP_TIMEOUT_SECONDS),
        (signal.SIGTERM, 5),
        (signal.SIGKILL, 5),
    ):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
    raise CollectorError(f"process group {process.pid} did not stop")


class ReferenceServer:
    def __init__(self, binary: Path, model: Path, root: Path, port: int) -> None:
        self.binary = binary
        self.model = model
        self.root = root
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.process: subprocess.Popen[bytes] | None = None
        self.stdout_handle: Any = None
        self.stderr_handle: Any = None

    def __enter__(self) -> ReferenceServer:
        self.root.mkdir(parents=True, exist_ok=True)
        argv = reference_server_argv(self.binary, self.model, self.port)
        atomic_write_json(self.root / "server.command.json", {"argv": argv})
        self.stdout_handle = (self.root / "server.stdout.log").open("wb")
        self.stderr_handle = (self.root / "server.stderr.log").open("wb")
        self.process = subprocess.Popen(
            argv,
            cwd=numerics.REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=self.stdout_handle,
            stderr=self.stderr_handle,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + SERVER_START_TIMEOUT_SECONDS
            next_progress = time.monotonic()
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise CollectorError(f"llama-server exited during startup with {self.process.returncode}")
                try:
                    health = http_json(self.base_url, "/health", None, timeout_seconds=2)
                    if health.get("status") == "ok":
                        atomic_write_json(self.root / "health.ready.json", health)
                        print(f"reference server ready: port={self.port}", flush=True)
                        return self
                except CollectorError:
                    pass
                now = time.monotonic()
                if now >= next_progress:
                    print(f"reference server loading: remaining={max(0, int(deadline - now))}s", flush=True)
                    next_progress = now + 10
                time.sleep(0.5)
            raise CollectorError(f"llama-server did not become healthy within {SERVER_START_TIMEOUT_SECONDS}s")
        except Exception:
            self._cleanup()
            raise

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        try:
            if self.process is not None:
                stop_process_group(self.process)
        finally:
            if self.stdout_handle is not None:
                self.stdout_handle.close()
            if self.stderr_handle is not None:
                self.stderr_handle.close()

    def post(self, path: str, payload: dict[str, Any], case_root: Path, name: str) -> dict[str, Any]:
        atomic_write_json(case_root / f"{name}.request.json", payload)
        response = http_json(
            self.base_url,
            path,
            payload,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        atomic_write_json(case_root / f"{name}.response.json", response)
        return response


def run_bounded_process(argv: list[str], root: Path) -> int:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("FERRUM_")}
    environment.update(FERRUM_ENV_LIMITS)
    environment["NO_COLOR"] = "1"
    process = subprocess.Popen(
        argv,
        cwd=numerics.REPO_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=REQUEST_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        stop_process_group(process)
        stdout, stderr = process.communicate()
        atomic_write(root / "stdout.jsonl", stdout)
        atomic_write(root / "stderr.log", stderr)
        raise CollectorError(f"Ferrum run exceeded {REQUEST_TIMEOUT_SECONDS}s") from error
    atomic_write(root / "stdout.jsonl", stdout)
    atomic_write(root / "stderr.log", stderr)
    return int(process.returncode)


def token_array(value: Any, label: str, expected_count: int | None) -> list[int]:
    require(isinstance(value, list) and value, f"{label} must be a non-empty array")
    require(
        all(isinstance(token, int) and not isinstance(token, bool) and 0 <= token < 2**32 for token in value),
        f"{label} contains an invalid token",
    )
    if expected_count is not None:
        require(len(value) == expected_count, f"{label} must contain exactly {expected_count} tokens")
    return list(value)


def find_request_bundle(dump_dir: Path) -> Path:
    candidates = sorted(
        path.parent
        for path in dump_dir.glob("*/prompt_token_ids.json")
        if (path.parent / "output_token_ids.json").is_file()
    )
    require(len(candidates) == 1, f"Ferrum request dump must contain exactly one complete bundle, found {len(candidates)}")
    return candidates[0]


def validate_ferrum_stdout(path: Path, prompt: str, request_id: str, output_tokens: list[int]) -> None:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise CollectorError(f"Ferrum stdout line {line_number} is not JSON") from error
        require(isinstance(value, dict), f"Ferrum stdout line {line_number} is not an object")
        records.append(value)
    require(records and all(row.get("schema_version") == 2 for row in records), "Ferrum JSONL schema differs")
    ready = [row for row in records if row.get("event") == "ready"]
    users = [row for row in records if row.get("event") == "user"]
    assistants = [row for row in records if row.get("event") == "assistant"]
    exits = [row for row in records if row.get("event") == "exit"]
    deltas = [row for row in records if row.get("event") == "assistant_delta"]
    require(
        len(ready) == len(users) == len(assistants) == len(exits) == 1,
        "Ferrum run must emit one ready/user/assistant/exit record",
    )
    require(
        users[0].get("content") == prompt
        and users[0].get("turn") == 0
        and users[0].get("request_id") == request_id
        and assistants[0].get("request_id") == request_id,
        "Ferrum JSONL request identity or prompt differs",
    )
    require(ready[0].get("session_id") == users[0].get("session_id") == assistants[0].get("session_id"), "Ferrum JSONL session identity differs")
    require(exits[0].get("reason") == "one_shot_complete", "Ferrum run did not use the one-shot product exit")
    emitted_tokens = [row.get("token_id") for row in deltas if isinstance(row.get("token_id"), int)]
    output_cursor = iter(output_tokens)
    for emitted in emitted_tokens:
        require(any(token == emitted for token in output_cursor), "Ferrum assistant_delta tokens are not an ordered output subsequence")
    usage = assistants[0].get("usage")
    require(
        assistants[0].get("n_tokens") == numerics.TOKEN_COUNT
        and isinstance(usage, dict)
        and usage.get("completion_tokens") == numerics.TOKEN_COUNT
        and assistants[0].get("finish_reason") == "length",
        "Ferrum run did not finish at the explicit 64-token ceiling",
    )


def collect_ferrum_case(binary: Path, prompt: str, case_root: Path) -> tuple[list[int], list[int]]:
    root = case_root / "ferrum"
    dump_dir = root / "request-dump"
    root.mkdir(parents=True, exist_ok=True)
    argv = ferrum_argv(binary, prompt, dump_dir)
    atomic_write_json(
        root / "command.json",
        {
            "argv": argv,
            "environment_overrides": {**FERRUM_ENV_LIMITS, "NO_COLOR": "1"},
            "removed_environment_keys": sorted(key for key in os.environ if key.startswith("FERRUM_")),
            "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        },
    )
    returncode = run_bounded_process(argv, root)
    require(returncode == 0, f"Ferrum run failed with exit code {returncode}; see {root / 'stderr.log'}")
    bundle = find_request_bundle(dump_dir)
    prompt_record = read_object(bundle / "prompt_token_ids.json", "Ferrum prompt tokens")
    output_record = read_object(bundle / "output_token_ids.json", "Ferrum output tokens")
    request_id = prompt_record.get("request_id")
    require(isinstance(request_id, str) and request_id, "Ferrum prompt request_id is invalid")
    require(output_record.get("request_id") == request_id, "Ferrum prompt/output request IDs differ")
    prompt_tokens = token_array(prompt_record.get("token_ids"), "Ferrum prompt tokens", None)
    output_tokens = token_array(
        output_record.get("token_ids"),
        "Ferrum output tokens",
        numerics.TOKEN_COUNT,
    )
    require(prompt_record.get("token_count") == len(prompt_tokens), "Ferrum prompt token_count differs")
    require(
        output_record.get("token_count") == numerics.TOKEN_COUNT
        and output_record.get("finish_reason") == "length",
        "Ferrum output dump did not finish at 64 tokens",
    )
    bad_output = read_object(bundle / "bad_output_scan.json", "Ferrum bad-output scan")
    require(bad_output.get("bad_output") is False, "Ferrum bad-output scan rejected generated text")
    validate_ferrum_stdout(root / "stdout.jsonl", prompt, request_id, output_tokens)
    atomic_write_json(
        root / "summary.json",
        {
            "request_bundle": str(bundle.resolve()),
            "prompt_token_count": len(prompt_tokens),
            "prompt_token_ids_sha256": numerics.token_sha256(prompt_tokens),
            "output_token_count": len(output_tokens),
            "output_token_ids_sha256": numerics.token_sha256(output_tokens),
        },
    )
    return prompt_tokens, output_tokens


def completion_request(prompt_tokens: list[int]) -> dict[str, Any]:
    return {
        "prompt": prompt_tokens,
        "n_predict": numerics.TOKEN_COUNT,
        "temperature": 0.0,
        "seed": 9271,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repeat_penalty": 1.0,
        "repeat_last_n": 64,
        "samplers": ["top_k", "top_p", "min_p", "temperature"],
        "return_tokens": True,
        "n_probs": 2,
        "post_sampling_probs": False,
        "cache_prompt": False,
        "stream": False,
        "ignore_eos": False,
    }


def parse_reference_probabilities(response: dict[str, Any], tokens: list[int]) -> tuple[list[float], list[int]]:
    rows = response.get("completion_probabilities")
    require(isinstance(rows, list) and len(rows) == numerics.TOKEN_COUNT, "llama.cpp probability row count differs")
    margins: list[float] = []
    near_ties: list[int] = []
    for step, (raw, selected) in enumerate(zip(rows, tokens, strict=True)):
        require(isinstance(raw, dict) and raw.get("id") == selected, f"llama.cpp probability token differs at step {step}")
        top = raw.get("top_logprobs")
        require(isinstance(top, list) and len(top) == 2, f"llama.cpp top-2 logits missing at step {step}")
        require(all(isinstance(row, dict) for row in top), f"llama.cpp top-2 row is invalid at step {step}")
        first_id = top[0].get("id")
        first = top[0].get("logprob")
        second = top[1].get("logprob")
        require(first_id == selected, f"llama.cpp top-1 token differs at step {step}")
        require(
            isinstance(first, (int, float))
            and isinstance(second, (int, float))
            and math.isfinite(float(first))
            and math.isfinite(float(second)),
            f"llama.cpp top-2 logits are non-finite at step {step}",
        )
        margin = float(first) - float(second)
        require(margin >= 0.0, f"llama.cpp top-2 order differs at step {step}")
        margins.append(margin)
        if margin < numerics.NEAR_TIE_MARGIN:
            near_ties.append(step)
    return margins, near_ties


def collect_reference_case(
    server: ReferenceServer,
    messages: list[dict[str, str]],
    case_root: Path,
) -> tuple[str, list[int], list[int], list[float], list[int]]:
    root = case_root / "reference"
    root.mkdir(parents=True, exist_ok=True)
    applied = server.post("/apply-template", {"messages": messages}, root, "apply-template")
    rendered = applied.get("prompt")
    require(isinstance(rendered, str) and rendered, "llama.cpp rendered prompt is empty")
    tokenized = server.post(
        "/tokenize",
        {"content": rendered, "add_special": False, "parse_special": True},
        root,
        "tokenize",
    )
    prompt_tokens = token_array(tokenized.get("tokens"), "llama.cpp prompt tokens", None)
    completed = server.post("/completion", completion_request(prompt_tokens), root, "completion")
    output_tokens = token_array(
        completed.get("tokens"),
        "llama.cpp output tokens",
        numerics.TOKEN_COUNT,
    )
    require(
        completed.get("stop") is True
        and completed.get("stop_type") == "limit"
        and completed.get("tokens_evaluated") == len(prompt_tokens),
        "llama.cpp completion did not finish at the explicit 64-token ceiling",
    )
    margins, near_ties = parse_reference_probabilities(completed, output_tokens)
    atomic_write_json(
        root / "summary.json",
        {
            "rendered_prompt_sha256": sha256_bytes(rendered.encode()),
            "prompt_token_count": len(prompt_tokens),
            "prompt_token_ids_sha256": numerics.token_sha256(prompt_tokens),
            "output_token_count": len(output_tokens),
            "output_token_ids_sha256": numerics.token_sha256(output_tokens),
            "near_tie_steps": near_ties,
        },
    )
    return rendered, prompt_tokens, output_tokens, margins, near_ties


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def case_cache_key(context: dict[str, Any], prompt: dict[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "source_git_sha": context["source_git_sha"],
                "source_tree_sha": context["source_tree_sha"],
                "ferrum_binary_sha256": context["ferrum_binary"]["sha256"],
                "reference_source_git_sha": context["reference_source_git_sha"],
                "reference_binary_sha256": context["reference_binary"]["sha256"],
                "model_file_sha256": context["model_file_sha256"],
                "prompt": prompt,
                "deterministic_config": context["deterministic_config"],
                "ferrum_command": context["command_contract"]["ferrum_argv_prefix"],
                "reference_command": context["command_contract"]["reference_argv_prefix"],
            }
        )
    )


def cached_case(path: Path, expected_key: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = read_object(path, "cached token parity case")
    if value.get("cache_key") != expected_key or not isinstance(value.get("case"), dict):
        return None
    case = value["case"]
    if set(case) != numerics.TOKEN_CASE_FIELDS or case.get("status") != "pass":
        return None
    return case


def collect_case(
    server: ReferenceServer,
    binary: Path,
    prompt: dict[str, Any],
    case_root: Path,
) -> dict[str, Any]:
    prompt_id = prompt.get("id")
    messages = prompt.get("messages")
    require(isinstance(prompt_id, str) and prompt_id, "prompt id is invalid")
    require(isinstance(messages, list) and messages, f"{prompt_id} messages are invalid")
    require(
        all(
            isinstance(message, dict)
            and set(message) == {"role", "content"}
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and message["content"]
            for message in messages
        ),
        f"{prompt_id} must contain non-empty user messages",
    )
    require(len(messages) == 1, f"{prompt_id} must contain exactly one user message")
    content = messages[0]["content"]
    ferrum_prompt, ferrum_output = collect_ferrum_case(binary, content, case_root)
    rendered, reference_prompt, reference_output, margins, near_ties = collect_reference_case(
        server,
        messages,
        case_root,
    )
    prompt_difference = first_difference(ferrum_prompt, reference_prompt)
    require(prompt_difference is None, f"{prompt_id} prompt token mismatch at index {prompt_difference}")
    output_difference = first_difference(ferrum_output, reference_output)
    return {
        "prompt_id": prompt_id,
        "prompt_sha256": numerics.canonical_sha256(messages),
        "rendered_prompt_sha256": sha256_bytes(rendered.encode()),
        "ferrum_prompt_token_ids": ferrum_prompt,
        "reference_prompt_token_ids": reference_prompt,
        "prompt_token_ids_sha256": numerics.token_sha256(ferrum_prompt),
        "ferrum_generated_token_ids": ferrum_output,
        "reference_generated_token_ids": reference_output,
        "ferrum_generated_token_ids_sha256": numerics.token_sha256(ferrum_output),
        "reference_generated_token_ids_sha256": numerics.token_sha256(reference_output),
        "first_generated_token_difference": output_difference,
        "shared_prefix_token_count": numerics.TOKEN_COUNT if output_difference is None else output_difference,
        "generated_sequences_equal": output_difference is None,
        "reference_top2_margins": margins,
        "reference_near_tie_steps": near_ties,
        "status": "pass",
    }


def command_contract(ferrum_binary: str, reference_binary: str) -> dict[str, Any]:
    return {
        "ferrum_entrypoint": "run",
        "ferrum_argv_prefix": ferrum_command_contract(ferrum_binary),
        "reference_entrypoint": "llama-server",
        "reference_argv_prefix": reference_server_contract(reference_binary),
        "reference_http_contract": {
            "apply_template_path": "/apply-template",
            "tokenize_path": "/tokenize",
            "completion_path": "/completion",
            "apply_template_count": numerics.PROMPT_COUNT,
            "tokenize_count": numerics.PROMPT_COUNT,
            "completion_count": numerics.PROMPT_COUNT,
            "tokenize_add_special": False,
            "tokenize_parse_special": True,
            "completion_prompt_kind": "exact_token_ids",
            "completion_request": {
                key: value
                for key, value in completion_request([]).items()
                if key != "prompt"
            },
        },
        "ferrum_execution_count": numerics.PROMPT_COUNT,
        "reference_execution_count": numerics.PROMPT_COUNT,
    }


def deterministic_config() -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "seed": 9271,
        "enable_thinking": False,
        "max_output_tokens": numerics.TOKEN_COUNT,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    }


def collection_context(args: argparse.Namespace) -> dict[str, Any]:
    source_git_sha, source_tree_sha, _catalog_blob, _summary = numerics.current_source_identity(require_clean=True)
    ferrum = binary_identity(args.ferrum_binary, "Ferrum binary")
    reference = binary_identity(args.llama_server_binary, "llama-server binary")
    llama_source = clean_git_identity(args.llama_cpp_source, "llama.cpp")
    identity = numerics.model_lock_identity()
    model = locked_model_path(
        args.model,
        identity["model_revision"],
        identity["model_file_sha256"],
    )
    return {
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "ferrum_binary": ferrum,
        "reference_source_git_sha": llama_source["git_sha"],
        "reference_binary": reference,
        "model": model,
        **identity,
        "deterministic_config": deterministic_config(),
        "command_contract": command_contract(ferrum["path"], reference["path"]),
    }


def validate_corpus() -> list[dict[str, Any]]:
    corpus = read_object(numerics.PROMPT_CORPUS, "token parity prompt corpus")
    prompts = corpus.get("prompts")
    require(
        corpus.get("schema_version") == 1
        and corpus.get("model_key") == numerics.MODEL_KEY
        and corpus.get("prompt_count") == numerics.PROMPT_COUNT
        and isinstance(prompts, list)
        and len(prompts) == numerics.PROMPT_COUNT,
        "token parity prompt corpus contract differs",
    )
    ids = [prompt.get("id") for prompt in prompts if isinstance(prompt, dict)]
    require(len(ids) == len(set(ids)) == numerics.PROMPT_COUNT, "token parity prompt IDs must be unique")
    return prompts


def build_parity(context: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "status": "pass",
        "source_git_sha": context["source_git_sha"],
        "source_tree_sha": context["source_tree_sha"],
        "source_dirty": False,
        "model_key": numerics.MODEL_KEY,
        "backend": numerics.BACKEND,
        "model_revision": context["model_revision"],
        "model_file_sha256": context["model_file_sha256"],
        "semantic_revision": context["semantic_revision"],
        "chat_template_sha256": context["chat_template_sha256"],
        "models_lock_sha256": context["models_lock_sha256"],
        "prompt_corpus_sha256": sha256_file(numerics.PROMPT_CORPUS),
        "reference_kind": "same_gguf_llama_cpp_external_free_run_diagnostic",
        "ferrum_binary": context["ferrum_binary"],
        "reference_source_git_sha": context["reference_source_git_sha"],
        "reference_source_dirty": False,
        "reference_binary": context["reference_binary"],
        "deterministic_config": context["deterministic_config"],
        "command_contract": context["command_contract"],
        "case_count": numerics.PROMPT_COUNT,
        "passed_count": numerics.PROMPT_COUNT,
        "exception_count": 0,
        "waiver_count": 0,
        "cases": cases,
    }


def collect(args: argparse.Namespace) -> Path:
    out = args.out.expanduser().resolve()
    require(not out.is_relative_to(numerics.REPO_ROOT), "token parity artifacts must be outside the source tree")
    context = collection_context(args)
    prompts = validate_corpus()
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        out / "collector.json",
        {
            "schema_version": 1,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "source_git_sha": context["source_git_sha"],
            "source_tree_sha": context["source_tree_sha"],
            "ferrum_binary": context["ferrum_binary"],
            "reference_source_git_sha": context["reference_source_git_sha"],
            "reference_binary": context["reference_binary"],
            "model_path": str(context["model"]),
            "model_file_sha256": context["model_file_sha256"],
            "prompt_corpus_sha256": sha256_file(numerics.PROMPT_CORPUS),
            "resource_limits": {
                "reference_parallel": 1,
                "reference_threads": THREAD_LIMIT,
                "ferrum_environment_overrides": FERRUM_ENV_LIMITS,
                "server_start_timeout_seconds": SERVER_START_TIMEOUT_SECONDS,
                "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            },
        },
    )
    pending: list[tuple[dict[str, Any], Path, str]] = []
    cases_by_id: dict[str, dict[str, Any]] = {}
    for prompt in prompts:
        prompt_id = str(prompt["id"])
        case_root = out / "cases" / prompt_id
        key = case_cache_key(context, prompt)
        cached = cached_case(case_root / "case.result.json", key)
        if cached is None:
            pending.append((prompt, case_root, key))
        else:
            cases_by_id[prompt_id] = cached
            print(f"token parity {prompt_id}: cached PASS", flush=True)

    if pending:
        port = args.port if args.port is not None else free_port()
        require(0 < port < 65536, "reference server port is invalid")
        with ReferenceServer(Path(context["reference_binary"]["path"]), context["model"], out / "reference-server", port) as server:
            for index, (prompt, case_root, key) in enumerate(pending, start=1):
                prompt_id = str(prompt["id"])
                print(f"token parity {prompt_id}: start ({index}/{len(pending)} pending)", flush=True)
                try:
                    case = collect_case(
                        server,
                        Path(context["ferrum_binary"]["path"]),
                        prompt,
                        case_root,
                    )
                except Exception as error:
                    atomic_write_json(
                        case_root / "case.failure.json",
                        {
                            "schema_version": 1,
                            "status": "reject",
                            "cache_key": key,
                            "prompt_id": prompt_id,
                            "error": str(error),
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    raise
                atomic_write_json(
                    case_root / "case.result.json",
                    {"schema_version": 1, "status": "pass", "cache_key": key, "case": case},
                )
                cases_by_id[prompt_id] = case
                if case["generated_sequences_equal"]:
                    result = "free-run exact 64/64"
                else:
                    result = f"free-run diverged at {case['first_generated_token_difference']}"
                print(f"token parity {prompt_id}: prompt PASS; {result}", flush=True)

    cases = [cases_by_id[str(prompt["id"])] for prompt in prompts]
    parity = build_parity(context, cases)
    parity_path = out / "token-parity.json"
    atomic_write_json(parity_path, parity)
    summary = numerics.validate_token_parity(parity_path, context["source_git_sha"], context["source_tree_sha"])
    collector = read_object(out / "collector.json", "collector manifest")
    collector.update(
        {
            "status": "pass",
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "case_count": len(cases),
            "prompt_token_match_count": summary["prompt_token_match_count"],
            "product_output_token_count_per_runtime": summary["product_output_token_count_per_runtime"],
            "generated_sequence_match_count": summary["generated_sequence_match_count"],
            "shared_prefix_token_count": summary["shared_prefix_token_count"],
            "reference_near_tie_step_count": summary["reference_near_tie_step_count"],
            "same_history_required_decision_count": len(cases) * numerics.TOKEN_COUNT,
            "token_parity": {"path": str(parity_path), "sha256": sha256_file(parity_path)},
        }
    )
    atomic_write_json(out / "collector.json", collector)
    return parity_path


def self_test() -> None:
    reference_contract = reference_server_contract("/tmp/llama-server")
    ferrum_contract = ferrum_command_contract("/tmp/ferrum")
    require(reference_contract.count("--parallel") == 1, "reference process cap is missing")
    require(reference_contract.count("--threads") == 1, "reference thread cap is missing")
    require("--request-dump-dir" in ferrum_contract, "Ferrum product evidence path is missing")
    tokens = list(range(numerics.TOKEN_COUNT))
    response = {
        "completion_probabilities": [
            {
                "id": token,
                "top_logprobs": [
                    {"id": token, "logprob": -0.1},
                    {"id": token + 1000, "logprob": -0.6},
                ],
            }
            for token in tokens
        ]
    }
    margins, near_ties = parse_reference_probabilities(response, tokens)
    require(margins == [0.5] * numerics.TOKEN_COUNT and near_ties == [], "top-2 parser differs")
    response["completion_probabilities"][3]["top_logprobs"][1]["logprob"] = -0.1005
    _, near_ties = parse_reference_probabilities(response, tokens)
    require(near_ties == [3], "near-tie detection differs")
    require(first_difference([1, 2, 3], [1, 2, 4]) == 2, "token difference detection differs")
    with tempfile.TemporaryDirectory(prefix="g08a-token-parity-selftest-") as temporary:
        root = Path(temporary)
        path = root / "nested" / "value.json"
        atomic_write_json(path, {"b": 2, "a": 1})
        require(path.read_bytes() == b'{"a":1,"b":2}\n', "canonical JSON output differs")
        blob = root / "blobs" / "model.gguf"
        atomic_write(blob, b"locked-model")
        snapshot = root / ("a" * 40) / "model.gguf"
        snapshot.parent.mkdir()
        snapshot.symlink_to(blob)
        require(
            locked_model_path(snapshot, "a" * 40, sha256_file(blob)) == blob.resolve(),
            "HF snapshot symlink validation differs",
        )
    print(SELFTEST_PASS)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--self-test", action="store_true")
    result.add_argument("--ferrum-binary", type=Path)
    result.add_argument("--llama-server-binary", type=Path)
    result.add_argument("--llama-cpp-source", type=Path)
    result.add_argument("--model", type=Path)
    result.add_argument("--out", type=Path)
    result.add_argument("--port", type=int)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        missing = [
            flag
            for flag, value in (
                ("--ferrum-binary", args.ferrum_binary),
                ("--llama-server-binary", args.llama_server_binary),
                ("--llama-cpp-source", args.llama_cpp_source),
                ("--model", args.model),
                ("--out", args.out),
            )
            if value is None
        ]
        require(not missing, "missing required arguments: " + ", ".join(missing))
        assert args.out is not None
        try:
            parity_path = collect(args)
        except Exception as error:
            out = args.out.expanduser().resolve()
            if not out.is_relative_to(numerics.REPO_ROOT):
                atomic_write_json(
                    out / "failure.json",
                    {
                        "schema_version": 1,
                        "status": "reject",
                        "error": str(error),
                        "failed_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            raise
        print(f"{PASS_PREFIX}: {parity_path.parent}")
        return 0
    except Exception as error:
        print(f"FERRUM RUNTIME VNEXT G08A TOKEN PARITY COLLECTOR FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
