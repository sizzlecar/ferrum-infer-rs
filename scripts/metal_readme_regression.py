#!/usr/bin/env python3
"""Metal README regression runner.

Runs the README Apple Silicon rows with content correctness, multi-turn,
concurrency throughput, and swap-state evidence. Unlike the old
docs/bench/macos-2026-05-02 shell scripts, this runner does not stop after the
first failed correctness check; it records the failure and continues collecting
performance evidence for the remaining rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "release"))

from openai_tool_call_regression import run_tool_call_regression

CHAT_CHECK_MAX_TOKENS = 256


@dataclass(frozen=True)
class Cell:
    concurrency: int
    prompts: int
    baseline_tps: float
    random_input_len: int = 16
    random_output_len: int = 64


@dataclass(frozen=True)
class ModelCase:
    key: str
    label: str
    gguf: str
    tokenizer: str
    moe: bool
    cells: tuple[Cell, ...]
    default_min_max_seqs: int
    default_max_max_seqs: int | None = None
    serve_args: tuple[str, ...] = ()
    unsafe_batch_probe: Cell | None = None


CASES = (
    ModelCase(
        key="llama31_8b",
        label="Llama-3.1-8B",
        gguf="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        tokenizer="Meta-Llama-3.1-8B-Instruct.tokenizer.json",
        moe=False,
        cells=(
            Cell(concurrency=1, prompts=8, baseline_tps=29.1),
            Cell(concurrency=8, prompts=24, baseline_tps=46.4),
            Cell(concurrency=16, prompts=32, baseline_tps=80.9),
        ),
        default_min_max_seqs=16,
        serve_args=("--greedy-argmax",),
    ),
    ModelCase(
        key="qwen3_8b",
        label="Qwen3-8B",
        gguf="Qwen3-8B-Q4_K_M.gguf",
        tokenizer="Qwen3-8B.tokenizer.json",
        moe=False,
        cells=(Cell(concurrency=16, prompts=32, baseline_tps=57.1),),
        default_min_max_seqs=16,
        serve_args=("--greedy-argmax",),
    ),
    ModelCase(
        key="qwen3_30b_a3b",
        label="Qwen3-30B-A3B",
        gguf="Qwen3-30B-A3B-Q4_K_M.gguf",
        tokenizer="Qwen3-30B-A3B.tokenizer.json",
        moe=True,
        cells=(Cell(concurrency=16, prompts=32, baseline_tps=72.5),),
        default_min_max_seqs=16,
    ),
)


def run_text(cmd: list[str], *, timeout: int = 30) -> str:
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    ).stdout


def swapusage() -> str:
    return run_text(["sysctl", "vm.swapusage"], timeout=5).strip()


def vm_stat_head() -> str:
    out = run_text(["vm_stat"], timeout=5)
    return "\n".join(out.splitlines()[:12])


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def post_json(url: str, payload: dict[str, Any], timeout: int = 120) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)


def get_url(url: str, timeout: int = 5) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return 0, str(e)


def start_server(
    ferrum_bin: Path,
    model_path: Path,
    case: ModelCase,
    port: int,
    out_dir: Path,
    *,
    prefix: str | None = None,
    extra_args: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    env = env or run_cli_env()
    artifact_prefix = prefix or case.key
    effective_config = out_dir / f"{artifact_prefix}.effective_config.json"
    decision_trace = out_dir / f"{artifact_prefix}.decision_trace.jsonl"
    cmd = [
        str(ferrum_bin),
        "serve",
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--effective-config-json",
        str(effective_config),
        "--decision-trace-jsonl",
        str(decision_trace),
        *case.serve_args,
        *extra_args,
    ]
    write(out_dir / f"{artifact_prefix}.server_cmd.json", json.dumps(cmd, indent=2))
    stdout = (out_dir / f"{artifact_prefix}.server.stdout").open("w", encoding="utf-8")
    stderr = (out_dir / f"{artifact_prefix}.server.stderr").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    return proc


def runtime_entries_by_key(config_path: Path) -> dict[str, dict[str, Any]]:
    if not config_path.is_file():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    entries = data.get("entries")
    if not isinstance(entries, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("key"), str):
            result[entry["key"]] = entry
    return result


def runtime_entry_usize(entries: dict[str, dict[str, Any]], key: str) -> int | None:
    value = (entries.get(key) or {}).get("effective_value")
    try:
        return int(str(value))
    except Exception:
        return None


def startup_config_summary(
    config_path: Path,
    min_max_seqs: int,
    max_max_seqs: int | None = None,
) -> dict[str, Any]:
    entries = runtime_entries_by_key(config_path)
    max_seqs = runtime_entry_usize(entries, "FERRUM_PAGED_MAX_SEQS")
    values = {
        key: entries.get(key)
        for key in [
            "FERRUM_PAGED_MAX_SEQS",
            "FERRUM_KV_CAPACITY",
            "FERRUM_MAX_BATCH",
            "FERRUM_MAX_BATCHED_TOKENS",
            "FERRUM_METAL_PAGED_KV",
            "FERRUM_MOE_HOST_TOPK",
        ]
        if key in entries
    }
    return {
        "config_path": str(config_path),
        "max_sequences": max_seqs,
        "min_required_max_sequences": min_max_seqs,
        "max_allowed_max_sequences": max_max_seqs,
        "passed": (
            isinstance(max_seqs, int)
            and max_seqs >= min_max_seqs
            and (max_max_seqs is None or max_seqs <= max_max_seqs)
        ),
        "values": values,
    }


def run_default_startup_probe(
    ferrum_bin: Path,
    model_path: Path,
    case: ModelCase,
    port: int,
    out_dir: Path,
) -> dict[str, Any]:
    effective_config = out_dir / f"{case.key}.default.effective_config.json"
    decision_trace = out_dir / f"{case.key}.default.decision_trace.jsonl"
    cmd = [
        str(ferrum_bin),
        "serve",
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--effective-config-json",
        str(effective_config),
        "--decision-trace-jsonl",
        str(decision_trace),
    ]
    write(out_dir / f"{case.key}.default.server_cmd.json", json.dumps(cmd, indent=2))
    stdout = (out_dir / f"{case.key}.default.server.stdout").open("w", encoding="utf-8")
    stderr = (out_dir / f"{case.key}.default.server.stderr").open("w", encoding="utf-8")
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=run_cli_env(),
            text=True,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        deadline = time.time() + 90
        while time.time() < deadline:
            if effective_config.is_file() and effective_config.stat().st_size > 0:
                break
            if proc.poll() is not None:
                break
            time.sleep(0.25)
        summary = startup_config_summary(
            effective_config,
            case.default_min_max_seqs,
            case.default_max_max_seqs,
        )
        summary["server_rc"] = proc.poll()
        summary["config_written"] = effective_config.is_file()
        return summary
    finally:
        stop_server(proc)
        stdout.close()
        stderr.close()


def run_cli_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in [
        "FERRUM_METAL_PAGED_KV",
        "FERRUM_PAGED_MAX_SEQS",
        "FERRUM_KV_CAPACITY",
        "FERRUM_MAX_BATCH",
        "FERRUM_MAX_BATCHED_TOKENS",
        "FERRUM_MOE_BATCHED",
        "FERRUM_MOE_BATCHED_DECODE",
        "FERRUM_MOE_BATCH_THRESHOLD",
    ]:
        env.pop(key, None)
    return env


def unsafe_moe_batch_env() -> dict[str, str]:
    env = run_cli_env()
    env.update(
        {
            "FERRUM_METAL_PAGED_KV": "1",
            "FERRUM_PAGED_MAX_SEQS": "4",
            "FERRUM_MAX_BATCH": "16",
            "FERRUM_MAX_BATCHED_TOKENS": "2048",
            "FERRUM_MOE_BATCHED": "1",
            "FERRUM_MOE_BATCHED_DECODE": "1",
            "FERRUM_MOE_BATCH_THRESHOLD": "2",
        }
    )
    return env


def run_cli_check(
    ferrum_bin: Path,
    model_path: Path,
    case: ModelCase,
    out_dir: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    prefix = f"{case.key}.run"
    env = run_cli_env()
    input_text = "\n".join(
        [
            "请记住暗号是蓝色月亮。只回答：已记住。",
            "刚才暗号是什么？只回答暗号。",
            "/bye",
            "",
        ]
    )
    cmd = [
        str(ferrum_bin),
        "run",
        str(model_path),
        "--backend",
        "metal",
        "--max-tokens",
        "512",
        "--system",
        "请用中文回答，不要输出代码。",
        "--output-format",
        "jsonl",
    ]
    result: dict[str, Any] = {
        "passed": False,
        "rc": None,
        "assistant": [],
        "jsonl_perf": {},
        "text_long": {},
    }
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        write(out_dir / f"{prefix}.stdout", stdout)
        write(out_dir / f"{prefix}.stderr", stderr)
        result.update({"rc": "timeout", "error": f"run timed out after {timeout_sec}s"})
        return result

    write(out_dir / f"{prefix}.stdin", input_text)
    write(out_dir / f"{prefix}.stdout", proc.stdout)
    write(out_dir / f"{prefix}.stderr", proc.stderr)
    result["rc"] = proc.returncode

    assistant: list[str] = []
    assistant_rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "assistant":
            assistant.append(str(row.get("content", "")))
            assistant_rows.append(row)
    token_total = sum(
        int(row.get("n_tokens", 0))
        for row in assistant_rows
        if isinstance(row.get("n_tokens", 0), (int, float))
    )
    ms_total = sum(
        float(row.get("ms", 0.0))
        for row in assistant_rows
        if isinstance(row.get("ms", 0.0), (int, float))
    )
    stderr_lower = proc.stderr.lower()
    joined = "\n".join(assistant).lower()
    no_abort = (
        "failed assertion" not in stderr_lower
        and "command encoder" not in stderr_lower
        and "panic" not in stderr_lower
        and "failed to render model chat template" not in stderr_lower
        and "<unk>" not in joined
        and "[pad" not in joined
    )
    first_ok = len(assistant) >= 1 and "记住" in assistant[0]
    second_ok = len(assistant) >= 2 and "蓝色月亮" in assistant[1]
    jsonl_tok_s = (token_total / (ms_total / 1000.0)) if ms_total > 0 else None
    jsonl_perf_ok = token_total > 0 and jsonl_tok_s is not None and jsonl_tok_s > 1.0
    passed = proc.returncode == 0 and no_abort and first_ok and second_ok and jsonl_perf_ok
    text_long = run_cli_text_long_check(
        ferrum_bin,
        model_path,
        case,
        out_dir,
        timeout_sec,
    )
    result.update(
        {
            "assistant": assistant,
            "jsonl_perf": {
                "completion_tokens": token_total,
                "ms": ms_total,
                "tok_s": jsonl_tok_s,
                "passed": jsonl_perf_ok,
            },
            "text_long": text_long,
            "no_abort": no_abort,
            "first_turn_ok": first_ok,
            "second_turn_ok": second_ok,
            "contains_secret": "蓝色月亮" in "\n".join(assistant),
            "passed": passed and bool(text_long.get("passed")),
        }
    )
    write(out_dir / f"{prefix}.verdict.txt", json.dumps(result, indent=2, ensure_ascii=False))
    return result


def run_cli_text_long_check(
    ferrum_bin: Path,
    model_path: Path,
    case: ModelCase,
    out_dir: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    prefix = f"{case.key}.run_text_long"
    input_text = "\n".join(
        [
            "你好",
            "你是谁？",
            "你会什么？",
            "Rust 是什么？请用中文简短回答。",
            "/bye",
            "",
        ]
    )
    cmd = [
        str(ferrum_bin),
        "run",
        str(model_path),
        "--backend",
        "metal",
        "--max-tokens",
        "256",
        "--system",
        "请用中文回答。",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=run_cli_env(),
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        write(out_dir / f"{prefix}.stdin", input_text)
        write(out_dir / f"{prefix}.stdout", stdout)
        write(out_dir / f"{prefix}.stderr", stderr)
        return {"passed": False, "rc": "timeout", "error": f"text run timed out after {timeout_sec}s"}

    write(out_dir / f"{prefix}.stdin", input_text)
    write(out_dir / f"{prefix}.stdout", proc.stdout)
    write(out_dir / f"{prefix}.stderr", proc.stderr)
    combined_lower = (proc.stdout + "\n" + proc.stderr).lower()
    no_abort = (
        "panic" not in combined_lower
        and "kv cache overflow" not in combined_lower
        and "failed assertion" not in combined_lower
        and "command encoder" not in combined_lower
        and "failed to render model chat template" not in combined_lower
        and "<unk>" not in combined_lower
        and "[pad" not in combined_lower
        and not re.search(r"(?m)^\s*</think>\s*$", proc.stdout)
    )
    token_lines = [
        line for line in proc.stderr.splitlines() if "tokens," in line and "tok/s" in line
    ]
    tok_s_values = []
    for line in token_lines:
        match = re.search(r",\s*([0-9]+(?:\.[0-9]+)?) tok/s", line)
        if match:
            tok_s_values.append(float(match.group(1)))
    perf_ok = len(tok_s_values) >= 4 and min(tok_s_values) > 1.0
    rust_ok = "rust" in combined_lower or "编程语言" in combined_lower
    result = {
        "passed": proc.returncode == 0 and no_abort and rust_ok and perf_ok,
        "rc": proc.returncode,
        "no_abort": no_abort,
        "rust_ok": rust_ok,
        "turn_stat_lines": len(token_lines),
        "tok_s_values": tok_s_values,
        "perf_ok": perf_ok,
        "contains_think": "<think>" in proc.stdout,
        "orphan_think_close": bool(re.search(r"(?m)^\s*</think>\s*$", proc.stdout)),
        "contains_unk": "<unk>" in combined_lower,
        "contains_pad": "[pad" in combined_lower,
    }
    write(out_dir / f"{prefix}.verdict.txt", json.dumps(result, indent=2, ensure_ascii=False))
    return result


def stop_server(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def wait_ready(port: int, proc: subprocess.Popen[str], out_path: Path) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + 180
    last_status = 0
    last_body = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            write(out_path, f"server exited rc={proc.returncode}\n{last_body}\n")
            return False
        last_status, last_body = get_url(url)
        if last_status == 200:
            write(out_path, last_body)
            return True
        time.sleep(1)
    write(out_path, f"timeout status={last_status}\n{last_body}\n")
    return False


def chat_check(port: int, case: ModelCase, out_dir: Path) -> dict[str, Any]:
    base = f"http://127.0.0.1:{port}/v1/chat/completions"
    result: dict[str, Any] = {}
    paris_payload = {
        "model": case.label,
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France? Reply with just the city name.",
            }
        ],
        "max_tokens": CHAT_CHECK_MAX_TOKENS,
        "stream": False,
    }
    status, body = post_json(base, paris_payload)
    write(out_dir / f"{case.key}.paris_payload.json", json.dumps(paris_payload, indent=2))
    write(out_dir / f"{case.key}.paris_response.json", body)
    content = ""
    finish_reason = None
    try:
        parsed = json.loads(body)
        content = parsed["choices"][0]["message"]["content"]
        finish_reason = parsed["choices"][0].get("finish_reason")
    except Exception:
        content = body[:300]
    paris_pass = status == 200 and finish_reason != "length" and "paris" in content.lower()
    result["paris"] = {
        "status": status,
        "passed": paris_pass,
        "content": content,
        "finish_reason": finish_reason,
    }
    write(
        out_dir / f"{case.key}.paris_verdict.txt",
        f"content={content}\nfinish_reason={finish_reason}\npassed={str(paris_pass).lower()}\n",
    )

    multiturn_payload = {
        "model": case.label,
        "messages": [
            {
                "role": "system",
                "content": "请用中文回答，不要输出代码。",
            },
            {
                "role": "user",
                "content": "请记住暗号是蓝色月亮。只回答：已记住。",
            },
            {"role": "assistant", "content": "已记住。"},
            {
                "role": "user",
                "content": "刚才暗号是什么？只回答暗号。",
            },
        ],
        "max_tokens": CHAT_CHECK_MAX_TOKENS,
        "stream": False,
    }
    status, body = post_json(base, multiturn_payload)
    write(
        out_dir / f"{case.key}.multiturn_payload.json",
        json.dumps(multiturn_payload, indent=2),
    )
    write(out_dir / f"{case.key}.multiturn_response.json", body)
    content = ""
    finish_reason = None
    try:
        parsed = json.loads(body)
        content = parsed["choices"][0]["message"]["content"]
        finish_reason = parsed["choices"][0].get("finish_reason")
    except Exception:
        content = body[:300]
    multiturn_pass = status == 200 and finish_reason != "length" and "蓝色月亮" in content
    result["multiturn"] = {
        "status": status,
        "passed": multiturn_pass,
        "content": content,
        "finish_reason": finish_reason,
    }
    write(
        out_dir / f"{case.key}.multiturn_verdict.txt",
        f"content={content}\nfinish_reason={finish_reason}\npassed={str(multiturn_pass).lower()}\n",
    )

    stream_payload = {
        "model": case.label,
        "messages": [
            {
                "role": "user",
                "content": "请用一句话打招呼，必须包含“你好”。",
            }
        ],
        "max_tokens": CHAT_CHECK_MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    stream_chunks = 0
    stream_content = ""
    stream_done_count = 0
    stream_raw_lines: list[str] = []
    stream_status = 0
    stream_error = ""
    req = urllib.request.Request(
        base,
        data=json.dumps(stream_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            stream_status = resp.status
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                stream_raw_lines.append(line)
                data = line[6:]
                if data == "[DONE]":
                    stream_done_count += 1
                    break
                stream_chunks += 1
                try:
                    row = json.loads(data)
                    stream_content += (
                        row.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                        or ""
                    )
                except Exception:
                    pass
    except Exception as e:
        stream_error = str(e)
    write(
        out_dir / f"{case.key}.stream_payload.json",
        json.dumps(stream_payload, indent=2, ensure_ascii=False),
    )
    write(out_dir / f"{case.key}.stream_response.txt", stream_content)
    write(out_dir / f"{case.key}.stream_raw.sse", "\n".join(stream_raw_lines) + "\n")
    stream_pass = (
        stream_status == 200
        and stream_chunks > 1
        and bool(stream_content)
        and stream_done_count == 1
    )
    result["stream"] = {
        "status": stream_status,
        "passed": stream_pass,
        "chunks": stream_chunks,
        "done_count": stream_done_count,
        "content_head": stream_content[:240],
        "error": stream_error,
    }
    write(
        out_dir / f"{case.key}.stream_verdict.txt",
        json.dumps(result["stream"], indent=2, ensure_ascii=False),
    )
    result["stateful_loop"] = stateful_loop_check(base, case, out_dir)
    return result


def repeated_prefix_run(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.lower())
    if len(compact) < 12:
        return False
    for width in range(1, min(16, len(compact) // 3) + 1):
        prefix = compact[:width]
        if prefix and compact.startswith(prefix * 3):
            return True
    return False


def stateful_loop_check(base: str, case: ModelCase, out_dir: Path) -> dict[str, Any]:
    prompts = [
        ("paris", "What is the capital of France? Reply with just the city name.", "paris"),
        ("tokyo", "What is the capital of Japan? Reply with just the city name.", "tokyo"),
        ("math4", "What is 2+2? Reply with just the number.", "4"),
        ("blue", "What color is a clear daytime sky? Reply with one word.", "blue"),
        ("math25", "What is 5 times 5? Reply with just the number.", "25"),
        ("rome", "What is the capital of Italy? Reply with just the city name.", "rome"),
    ]
    rows: list[dict[str, Any]] = []
    for name, prompt, expected in prompts:
        payload = {
            "model": case.label,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "stream": False,
        }
        status, body = post_json(base, payload, timeout=120)
        content = ""
        finish_reason = None
        try:
            parsed = json.loads(body)
            content = parsed["choices"][0]["message"].get("content") or ""
            finish_reason = parsed["choices"][0].get("finish_reason")
        except Exception:
            content = body[:300]
        lower = content.lower()
        row = {
            "name": name,
            "status": status,
            "expected": expected,
            "expected_ok": expected in lower,
            "finish_reason": finish_reason,
            "length_finish": finish_reason == "length",
            "repeated_prefix": repeated_prefix_run(content),
            "content_head": content[:240],
        }
        row["passed"] = (
            row["status"] == 200
            and row["expected_ok"] is True
            and row["length_finish"] is False
            and row["repeated_prefix"] is False
        )
        rows.append(row)
    result = {
        "passed": all(bool(row.get("passed")) for row in rows),
        "requests": len(rows),
        "failed": sum(1 for row in rows if not row.get("passed")),
        "length_finishes": sum(1 for row in rows if row.get("length_finish") is True),
        "repeated_prefixes": sum(1 for row in rows if row.get("repeated_prefix") is True),
        "rows": rows,
    }
    write(
        out_dir / f"{case.key}.stateful_loop_verdict.txt",
        json.dumps(result, indent=2, ensure_ascii=False),
    )
    return result


def run_quality_cell(
    port: int,
    case: ModelCase,
    cell: Cell,
    out_dir: Path,
    *,
    artifact_prefix: str | None = None,
) -> dict[str, Any]:
    prefix = artifact_prefix or f"{case.key}.c{cell.concurrency}.quality"
    base = f"http://127.0.0.1:{port}/v1/chat/completions"
    nonce = f"M{cell.concurrency:02d}"

    def marker_line_ok(line: str, marker: str) -> bool:
        return line.strip().rstrip(".。!！,，;；:：") == marker

    def answer_line_ok(line: str, answer: str) -> bool:
        return line.strip().rstrip(".。!！,，;；:：") == answer

    def call(i: int) -> dict[str, Any]:
        marker = f"K{nonce}{i:02d}Z"
        answer = f"S{(i + 1) * (i + 1):04d}"
        payload = {
            "model": case.label,
            "temperature": 0,
            "max_tokens": 96,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Reply with exactly two short lines. "
                        f"Line 1 must be this code exactly: {marker}. "
                        f"Line 2 must be this checksum exactly: {answer}."
                    ),
                }
            ],
        }
        status, body = post_json(base, payload, timeout=180)
        content = ""
        finish_reason = None
        try:
            parsed = json.loads(body)
            content = parsed["choices"][0]["message"].get("content") or ""
            finish_reason = parsed["choices"][0].get("finish_reason")
        except Exception:
            content = body[:500]
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        format_ok = (
            len(lines) == 2
            and marker_line_ok(lines[0], marker)
            and answer_line_ok(lines[1], answer)
            and not any(other.startswith("K") and other.endswith("Z") for other in lines[1:])
        )
        return {
            "i": i,
            "status": status,
            "marker": marker,
            "square": answer,
            "marker_ok": marker in content,
            "square_ok": answer in content,
            "format_ok": format_ok,
            "finish_reason": finish_reason,
            "content_head": content[:240],
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cell.concurrency) as executor:
        futures = [executor.submit(call, i) for i in range(cell.concurrency)]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: int(row.get("i") or 0))

    markers = {row["i"]: row["marker"] for row in rows}
    crosstalk = 0
    for row in rows:
        content = str(row.get("content_head") or "")
        for j, marker in markers.items():
            if j != row["i"] and marker in content:
                crosstalk += 1

    status_ok = sum(1 for row in rows if row.get("status") == 200)
    marker_ok = sum(1 for row in rows if row.get("marker_ok") is True)
    square_ok = sum(1 for row in rows if row.get("square_ok") is True)
    format_ok = sum(1 for row in rows if row.get("format_ok") is True)
    length_finishes = sum(1 for row in rows if row.get("finish_reason") == "length")
    result = {
        "concurrency": cell.concurrency,
        "requests": cell.concurrency,
        "status_200": status_ok,
        "marker_ok": marker_ok,
        "square_ok": square_ok,
        "format_ok": format_ok,
        "crosstalk": crosstalk,
        "length_finishes": length_finishes,
        "passed": (
            status_ok == cell.concurrency
            and marker_ok == cell.concurrency
            and square_ok == cell.concurrency
            and format_ok == cell.concurrency
            and crosstalk == 0
            and length_finishes == 0
        ),
        "rows": rows,
    }
    write(out_dir / f"{prefix}.json", json.dumps(result, indent=2, ensure_ascii=False))
    return result


def run_tool_call_check(port: int, case: ModelCase, out_dir: Path) -> dict[str, Any]:
    out = out_dir / f"{case.key}.tool-call-regression"
    try:
        return run_tool_call_regression(f"http://127.0.0.1:{port}", case.label, out)
    except Exception as e:
        result = {"status": "fail", "model": case.label, "error": str(e)}
        write(
            out / "tool_call_regression.json",
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        )
        return result


def run_unsafe_moe_batch_probe(
    ferrum_bin: Path,
    model_path: Path,
    case: ModelCase,
    port: int,
    out_dir: Path,
) -> dict[str, Any]:
    cell = case.unsafe_batch_probe
    if not case.moe or cell is None:
        return {"enabled": False, "reason": "not a MoE unsafe batch probe target"}

    prefix = f"{case.key}.unsafe_batch"
    proc: subprocess.Popen[str] | None = None
    try:
        proc = start_server(
            ferrum_bin,
            model_path,
            case,
            port,
            out_dir,
            prefix=prefix,
            extra_args=(
                "--max-num-seqs",
                str(cell.concurrency),
                "--max-num-batched-tokens",
                "2048",
            ),
            env=unsafe_moe_batch_env(),
        )
        ready = wait_ready(port, proc, out_dir / f"{prefix}.models.json")
        startup = startup_config_summary(
            out_dir / f"{prefix}.effective_config.json",
            cell.concurrency,
        )
        quality = (
            run_quality_cell(
                port,
                case,
                cell,
                out_dir,
                artifact_prefix=f"{prefix}.c{cell.concurrency}.quality",
            )
            if ready
            else {"passed": False, "error": "server not ready"}
        )
        result = {
            "enabled": True,
            "product_default": False,
            "expected_default_safe_max_sequences": 1,
            "server_ready": ready,
            "startup": startup,
            "quality": {k: v for k, v in quality.items() if k != "rows"},
            "quality_rows_path": str(out_dir / f"{prefix}.c{cell.concurrency}.quality.json"),
        }
        write(out_dir / f"{prefix}.verdict.json", json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except Exception as e:
        result = {"enabled": True, "product_default": False, "error": str(e)}
        write(out_dir / f"{prefix}.verdict.json", json.dumps(result, indent=2, ensure_ascii=False))
        return result
    finally:
        stop_server(proc)
        time.sleep(4)


def run_bench_cell(
    ferrum_bin: Path,
    port: int,
    case: ModelCase,
    cell: Cell,
    out_dir: Path,
    tokenizers_dir: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    prefix = f"{case.key}.c{cell.concurrency}"
    result_file = out_dir / f"{prefix}.json"
    log_file = out_dir / f"{prefix}.bench.log"
    write(out_dir / f"{prefix}.swap_before.txt", swapusage() + "\n")
    token_dir = out_dir / f"{case.key}.tokenizer"
    token_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tokenizers_dir / case.tokenizer, token_dir / "tokenizer.json")
    cmd = [
        str(ferrum_bin),
        "bench-serve",
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--model",
        case.label,
        "--tokenizer",
        str(token_dir),
        "--num-prompts",
        str(cell.prompts),
        "--concurrency",
        str(cell.concurrency),
        "--random-input-len",
        str(cell.random_input_len),
        "--random-output-len",
        str(cell.random_output_len),
        "--warmup-requests",
        "1",
        "--n-repeats",
        "1",
        "--fail-on-error",
        "--seed",
        "9271",
        "--output",
        "json",
        "--out",
        str(result_file),
    ]
    row: dict[str, Any] = {
        "concurrency": cell.concurrency,
        "prompts": cell.prompts,
        "baseline_tps": cell.baseline_tps,
        "random_input_len": cell.random_input_len,
        "random_output_len": cell.random_output_len,
        "quality": run_quality_cell(port, case, cell, out_dir),
    }
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )
        write(log_file, proc.stdout)
        row["runner_rc"] = proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        write(
            log_file,
            stdout
            + f"\n[metal-readme-regression] timed out after {timeout_sec}s\n",
        )
        row.update(
            {
                "runner_rc": "timeout",
                "error": f"bench timed out after {timeout_sec}s",
                "completed": None,
                "failed": None,
                "output_throughput_tok_s": None,
                "ratio_to_readme": None,
                "not_regressed_90pct": False,
            }
        )
        return row
    finally:
        write(out_dir / f"{prefix}.swap_after.txt", swapusage() + "\n")
    if result_file.is_file():
        data = json.loads(result_file.read_text(encoding="utf-8"))
        tps_stat = data.get("output_throughput_tps") or {}
        tps = tps_stat.get("mean") if isinstance(tps_stat, dict) else None
        completed = sum(data.get("completed_per_run") or [])
        failed = sum(data.get("errored_per_run") or [])
        ratio = tps / cell.baseline_tps if isinstance(tps, (int, float)) else None
        row.update(
            {
                "completed": completed,
                "failed": failed,
                "output_throughput_tok_s": tps,
                "ratio_to_readme": ratio,
                "not_regressed_90pct": bool(
                    isinstance(ratio, float)
                    and ratio >= 0.90
                    and completed == cell.prompts
                    and failed == 0
                ),
                "ttft_median_ms": data.get("ttft_ms", {}).get("median"),
                "tpot_median_ms": data.get("tpot_ms", {}).get("median"),
            }
        )
    return row


def markdown_summary(report: dict[str, Any]) -> str:
    out = [f"# Metal README Regression - {date.today().isoformat()}", ""]
    out.append("Scope: README Apple Silicon rows with correctness, multi-turn, concurrency throughput, and swap evidence.")
    out.append("")
    out.append(f"Ferrum: `{report['ferrum_version'].strip()}`")
    out.append(f"Swap at start: `{report['swap_start']}`")
    out.append(f"Swap at end: `{report['swap_end']}`")
    out.append("")
    out.append("| Model | Default max seqs | Bench max seqs | Serve correctness | Serve multi-turn | Serve stream | Stateful loop | Tool call | Run REPL multi-turn | c | Quality | in/out | README tok/s | Current tok/s | Ratio | Completed | Gate |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for model in report["models"]:
        correctness = "pass" if model["chat"]["paris"]["passed"] else "FAIL"
        multiturn = "pass" if model["chat"]["multiturn"]["passed"] else "FAIL"
        stream_gate = "pass" if model["chat"].get("stream", {}).get("passed") else "FAIL"
        stateful_gate = "pass" if model["chat"].get("stateful_loop", {}).get("passed") else "FAIL"
        tool_gate = "pass" if model.get("tool_call", {}).get("status") == "pass" else "FAIL"
        run_gate = "pass" if model.get("run", {}).get("passed") else "FAIL"
        default_max = model.get("default_startup", {}).get("max_sequences")
        bench_max = model.get("serve_startup", {}).get("max_sequences")
        for cell in model["cells"]:
            tps = cell.get("output_throughput_tok_s")
            ratio = cell.get("ratio_to_readme")
            gate = "pass" if cell.get("not_regressed_90pct") else "FAIL"
            completed = f"{cell.get('completed')}/{cell.get('prompts')}"
            workload = f"{cell.get('random_input_len')}/{cell.get('random_output_len')}"
            quality = "pass" if (cell.get("quality") or {}).get("passed") else "FAIL"
            if isinstance(tps, (int, float)):
                out.append(
                    f"| {model['label']} | {default_max} | {bench_max} | {correctness} | {multiturn} | {stream_gate} | {stateful_gate} | {tool_gate} | {run_gate} | {cell['concurrency']} | {quality} | "
                    f"{workload} | {cell['baseline_tps']:.1f} | {tps:.1f} | {ratio:.3f} | {completed} | {gate} |"
                )
            else:
                out.append(
                    f"| {model['label']} | {default_max} | {bench_max} | {correctness} | {multiturn} | {stream_gate} | {stateful_gate} | {tool_gate} | {run_gate} | {cell['concurrency']} | {quality} | "
                    f"{workload} | {cell['baseline_tps']:.1f} | n/a | n/a | {completed} | {gate} |"
                )
    out.append("")
    out.append("Correctness prompts:")
    for model in report["models"]:
        perf = model.get("run", {}).get("jsonl_perf", {})
        out.append(
            f"- `{model['label']}` Paris: `{model['chat']['paris']['content']}`; "
            f"serve multi-turn: `{model['chat']['multiturn']['content']}`; "
            f"serve stream chunks: `{model['chat'].get('stream', {}).get('chunks')}`; "
            f"run multi-turn: `{model.get('run', {}).get('assistant', [])}`; "
            f"run tok/s: `{perf.get('tok_s')}`; "
            f"run text long: `{model.get('run', {}).get('text_long', {})}`"
        )
    out.append("")
    out.append("Notes:")
    out.append("- Performance gate is `current >= 0.90 * README baseline`, plus all requests completed.")
    out.append("- Default startup config must be captured without benchmark CLI overrides and must expose enough sequence slots for the release cell.")
    out.append("- Throughput-profile startup config must expose enough sequence slots for the measured concurrency cell.")
    out.append("- The stateful loop probe sends multiple short prompts through one server process and rejects repeated-prefix/length regressions.")
    out.append("- Every throughput cell first runs a marker/square concurrent quality probe; HTTP 200 and zero request errors are not sufficient correctness evidence.")
    out.append("- Metal MoE release evidence requires a multi-sequence content-quality and throughput cell.")
    out.append("- Throughput cells use canonical `ferrum bench-serve` with streaming usage token accounting.")
    out.append("- Throughput cells record their input/output token workload and the server CLI profile in the artifact directory.")
    out.append("- `run` coverage includes JSONL multi-turn plus default text-mode long multi-turn to catch streaming and KV overflow regressions.")
    out.append("- This runner records correctness failures and still collects performance data.")
    out.append("- Active swap means results are release-regression evidence, not clean marketing numbers.")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ferrum-bin", type=Path, default=ROOT / "target/release/ferrum")
    parser.add_argument("--models-dir", type=Path, default=Path("/Users/chejinxuan/ferrum-bench/models"))
    parser.add_argument("--tokenizers-dir", type=Path, default=Path("/Users/chejinxuan/ferrum-bench/tokenizers"))
    parser.add_argument("--port", type=int, default=18181)
    parser.add_argument("--bench-timeout-sec", type=int, default=300)
    parser.add_argument("--run-timeout-sec", type=int, default=300)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only matching model key(s), e.g. --only qwen3_30b_a3b. May be repeated.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "ferrum_version": run_text([str(args.ferrum_bin), "--version"], timeout=10),
        "swap_start": swapusage(),
        "vm_stat_start": vm_stat_head(),
        "models": [],
    }

    selected = [case for case in CASES if not args.only or case.key in set(args.only)]
    for case in selected:
        model_path = args.models_dir / case.gguf
        model_report: dict[str, Any] = {
            "key": case.key,
            "label": case.label,
            "gguf": str(model_path),
            "moe": case.moe,
            "default_startup": {},
            "serve_startup": {},
            "chat": {},
            "run": {},
            "cells": [],
        }
        proc: subprocess.Popen[str] | None = None
        try:
            model_report["default_startup"] = run_default_startup_probe(
                args.ferrum_bin,
                model_path,
                case,
                args.port + 100,
                args.out,
            )
            write(args.out / f"{case.key}.swap_before_server.txt", swapusage() + "\n")
            proc = start_server(args.ferrum_bin, model_path, case, args.port, args.out)
            ready = wait_ready(args.port, proc, args.out / f"{case.key}.models.json")
            model_report["server_ready"] = ready
            model_report["serve_startup"] = startup_config_summary(
                args.out / f"{case.key}.effective_config.json",
                max(cell.concurrency for cell in case.cells),
            )
            if ready:
                model_report["chat"] = chat_check(args.port, case, args.out)
                model_report["tool_call"] = run_tool_call_check(args.port, case, args.out)
                model_report["unsafe_batch_probe"] = run_unsafe_moe_batch_probe(
                    args.ferrum_bin,
                    model_path,
                    case,
                    args.port + 200,
                    args.out,
                )
                for cell in case.cells:
                    model_report["cells"].append(
                        run_bench_cell(
                            args.ferrum_bin,
                            args.port,
                            case,
                            cell,
                            args.out,
                            args.tokenizers_dir,
                            args.bench_timeout_sec,
                        )
                    )
            else:
                model_report["chat"] = {
                    "paris": {"passed": False, "content": "server not ready"},
                    "multiturn": {"passed": False, "content": "server not ready"},
                }
        except Exception as e:
            model_report["error"] = repr(e)
            if not model_report["chat"]:
                model_report["chat"] = {
                    "paris": {"passed": False, "content": repr(e)},
                    "multiturn": {"passed": False, "content": repr(e)},
                }
        finally:
            stop_server(proc)
            write(args.out / f"{case.key}.swap_after_server.txt", swapusage() + "\n")
            time.sleep(8)
        model_report["run"] = run_cli_check(
            args.ferrum_bin,
            model_path,
            case,
            args.out,
            args.run_timeout_sec,
        )
        report["models"].append(model_report)
        write(args.out / "summary.json", json.dumps(report, indent=2, ensure_ascii=False))

    report["swap_end"] = swapusage()
    report["vm_stat_end"] = vm_stat_head()
    write(args.out / "summary.json", json.dumps(report, indent=2, ensure_ascii=False))
    write(args.out / "summary.md", markdown_summary(report))
    print(args.out / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
