#!/usr/bin/env python3
"""G1 vLLM migration compatibility release gate.

This gate is intentionally CPU/Metal-local by default. It verifies that the
vLLM-compatible serve flags are product-visible, reflected in the effective
runtime config, and compatible with OpenAI-style streaming clients.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

BAD_LOG_PATTERNS = [
    "panicked",
    "panic",
    "KV cache overflow",
    "failed to render model chat template",
    "command encoder",
    "failed assertion",
    "<unk>",
    "[PAD]",
]

REQUIRED_FLAGS = [
    "--max-model-len",
    "--max-num-seqs",
    "--max-num-batched-tokens",
    "--enable-prefix-caching",
    "--no-enable-prefix-caching",
]

REQUIRED_CONFIG = {
    "FERRUM_MAX_MODEL_LEN": "2048",
    "FERRUM_PAGED_MAX_SEQS": "4",
    "FERRUM_MAX_BATCHED_TOKENS": "2048",
    "FERRUM_PREFIX_CACHE": "0",
}


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


def run(cmd: list[str], out: Path, log: GateLog, *, timeout: int = 600, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    log.write("RUN " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env={**os.environ, **(env or {})},
        check=False,
    )
    out.write_text(proc.stdout, errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(cmd)}; log={out}")
    assert_no_bad_patterns(out.name, proc.stdout)
    return proc


def git_value(args: list[str], default: str = "unknown") -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=repo_root(), text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        value = proc.stdout.strip()
        return value or default
    except Exception:
        return default


def default_out_dir() -> Path:
    short = git_value(["rev-parse", "--short", "HEAD"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return repo_root() / "docs" / "release" / "g1-g4" / "g1-vllm-migration" / f"{stamp}-{short}"


def assert_no_bad_patterns(label: str, text: str) -> None:
    lower = text.lower()
    for pattern in BAD_LOG_PATTERNS:
        if pattern.lower() in lower:
            raise RuntimeError(f"forbidden pattern {pattern!r} in {label}")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request_json(url: str, payload: dict[str, Any] | None = None, *, timeout: int = 120) -> tuple[int, str]:
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", "replace")


def wait_health(base_url: str, timeout: int, log: GateLog) -> None:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            status, body = request_json(base_url + "/health", None, timeout=2)
            if status == 200:
                log.write("health OK")
                return
            last = f"status={status} body={body[:200]}"
        except Exception as exc:  # noqa: BLE001 - gate should keep polling until deadline.
            last = repr(exc)
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy within {timeout}s; last={last}")


def parse_sse(body: str) -> tuple[list[dict[str, Any]], int, list[str]]:
    chunks: list[dict[str, Any]] = []
    done_count = 0
    sequence: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ").strip()
        if data == "[DONE]":
            done_count += 1
            sequence.append("done")
            continue
        if not data:
            continue
        parsed = json.loads(data)
        chunks.append(parsed)
        if isinstance(parsed.get("usage"), dict) and parsed.get("choices") == []:
            sequence.append("usage")
        else:
            sequence.append("chunk")
    return chunks, done_count, sequence


def check_effective_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    entries = data.get("entries", [])
    by_key = {entry.get("key"): entry for entry in entries}
    checked: dict[str, Any] = {}
    for key, expected in REQUIRED_CONFIG.items():
        entry = by_key.get(key)
        if not entry:
            raise RuntimeError(f"effective config missing {key}: {path}")
        value = str(entry.get("effective_value"))
        source = str(entry.get("source"))
        if value != expected:
            raise RuntimeError(f"effective config {key}={value}, expected {expected}")
        if source.lower() != "cli":
            raise RuntimeError(f"effective config {key} source={source}, expected cli")
        checked[key] = {"effective_value": value, "source": source}
    return checked


def require_nonempty_chat(label: str, body: str) -> str:
    data = json.loads(body)
    content = data["choices"][0]["message"].get("content") or ""
    assert_no_bad_patterns(label, body)
    if not content.strip():
        raise RuntimeError(f"empty chat content for {label}: {body[:500]}")
    if int(data.get("usage", {}).get("total_tokens", 0)) <= 0:
        raise RuntimeError(f"missing usage tokens for {label}: {body[:500]}")
    return content


def direct_smoke(bin_path: Path, model: str, model_name: str, out: Path, log: GateLog, port: int | None) -> dict[str, Any]:
    port = port or free_port()
    base = f"http://127.0.0.1:{port}"
    effective_config = out / "effective-config.json"
    serve_log = out / "serve.log"
    cmd = [
        str(bin_path),
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        REQUIRED_CONFIG["FERRUM_MAX_MODEL_LEN"],
        "--max-num-seqs",
        REQUIRED_CONFIG["FERRUM_PAGED_MAX_SEQS"],
        "--max-num-batched-tokens",
        REQUIRED_CONFIG["FERRUM_MAX_BATCHED_TOKENS"],
        "--no-enable-prefix-caching",
        "--effective-config-json",
        str(effective_config),
    ]
    log.write("START " + " ".join(cmd))
    env = {**os.environ, "NO_COLOR": "1"}
    with serve_log.open("wb") as log_file:
        proc = subprocess.Popen(cmd, cwd=repo_root(), stdout=log_file, stderr=subprocess.STDOUT, env=env)
    try:
        wait_health(base, 180, log)
        config_checked = check_effective_config(effective_config)

        status, models_body = request_json(base + "/v1/models", None)
        (out / "models.json").write_text(models_body, errors="replace")
        if status != 200 or json.loads(models_body).get("object") != "list":
            raise RuntimeError(f"/v1/models failed status={status}: {models_body[:500]}")

        chat_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Say hi in one short sentence."}],
            "temperature": 0,
            "max_tokens": 512,
        }
        status, chat_body = request_json(base + "/v1/chat/completions", chat_payload)
        (out / "chat.json").write_text(chat_body, errors="replace")
        if status != 200:
            raise RuntimeError(f"chat failed status={status}: {chat_body[:500]}")
        chat_content = require_nonempty_chat("chat.json", chat_body)

        stream_payload = {**chat_payload, "stream": True}
        status, stream_body = request_json(base + "/v1/chat/completions", stream_payload)
        (out / "stream.sse").write_text(stream_body, errors="replace")
        if status != 200:
            raise RuntimeError(f"stream failed status={status}: {stream_body[:500]}")
        stream_chunks, done_count, stream_sequence = parse_sse(stream_body)
        stream_content = "".join(
            chunk.get("choices", [{}])[0].get("delta", {}).get("content") or ""
            for chunk in stream_chunks
            if chunk.get("choices")
        )
        if done_count != 1 or not stream_chunks or not stream_content.strip():
            raise RuntimeError(f"bad stream done={done_count} chunks={len(stream_chunks)} content={stream_content!r}")

        usage_payload = {**chat_payload, "stream": True, "stream_options": {"include_usage": True}}
        status, usage_body = request_json(base + "/v1/chat/completions", usage_payload)
        (out / "usage_stream.sse").write_text(usage_body, errors="replace")
        if status != 200:
            raise RuntimeError(f"usage stream failed status={status}: {usage_body[:500]}")
        usage_chunks, usage_done_count, usage_sequence = parse_sse(usage_body)
        usage_count = usage_sequence.count("usage")
        if usage_done_count != 1 or usage_count != 1:
            raise RuntimeError(f"bad usage SSE done={usage_done_count} usage={usage_count} seq={usage_sequence}")
        if usage_sequence.index("usage") > usage_sequence.index("done"):
            raise RuntimeError(f"usage chunk must appear before [DONE]: {usage_sequence}")
        if not any((chunk.get("usage") or {}).get("total_tokens") for chunk in usage_chunks):
            raise RuntimeError("usage SSE did not include total_tokens")

        serve_text = serve_log.read_text(errors="replace") if serve_log.exists() else ""
        assert_no_bad_patterns("serve.log", serve_text)

        return {
            "passed": True,
            "base_url": base,
            "model": model,
            "model_name": model_name,
            "effective_config": config_checked,
            "chat_content_len": len(chat_content),
            "stream_chunks": len(stream_chunks),
            "stream_done_count": done_count,
            "usage_stream_chunks": len(usage_chunks),
            "usage_done_count": usage_done_count,
            "usage_chunk_count": usage_count,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        serve_text = serve_log.read_text(errors="replace") if serve_log.exists() else ""
        assert_no_bad_patterns("serve.log", serve_text)


def write_manifest(out: Path, args: argparse.Namespace, checks: dict[str, Any]) -> None:
    manifest = {
        "goal": "g1-vllm-migration",
        "status": "pass",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo": {
            "head": git_value(["rev-parse", "HEAD"]),
            "short_head": git_value(["rev-parse", "--short", "HEAD"]),
            "branch": git_value(["branch", "--show-current"]),
            "dirty": bool(git_value(["status", "--porcelain"], "")),
        },
        "model": args.model,
        "model_name": args.model_name,
        "checks": checks,
        "artifacts": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (out / "summary.md").write_text(
        "# G1 vLLM Migration Compatibility Gate\n\n"
        f"Status: PASS\n\n"
        f"Model: `{args.model}`\n\n"
        "Validated:\n"
        "- vLLM-compatible `ferrum serve` flags are visible in help output.\n"
        "- CLI flags are reflected in effective runtime config with `cli` source.\n"
        "- `/v1/models`, non-stream chat, stream chat, and usage stream smoke passed.\n"
        "- Streaming emitted exactly one `[DONE]`; usage stream emitted exactly one usage chunk before `[DONE]`.\n"
        "- Server log was scanned for forbidden release-blocker patterns.\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="artifact directory")
    parser.add_argument("--model", default="qwen3:0.6b", help="local Ferrum model path or alias")
    parser.add_argument("--model-name", default=None, help="model field sent to OpenAI API; defaults to --model")
    parser.add_argument("--ferrum-bin", type=Path, default=repo_root() / "target" / "release" / "ferrum")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--skip-workspace-test", action="store_true", help="skip cargo test --workspace --all-targets")
    parser.add_argument("--skip-ignored-rust-smoke", action="store_true", help="skip ignored Rust smoke that loads the same model")
    args = parser.parse_args()
    args.model_name = args.model_name or args.model

    out = (args.out or default_out_dir()).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = GateLog(out / "gate.log")

    checks: dict[str, Any] = {}
    if not args.skip_workspace_test:
        run(["cargo", "test", "--workspace", "--all-targets"], out / "cargo-test.log", log, timeout=2400)
        checks["cargo_workspace_all_targets"] = True
    else:
        (out / "cargo-test.log").write_text("skipped by --skip-workspace-test\n")
        checks["cargo_workspace_all_targets"] = "skipped"

    run(
        ["cargo", "test", "-p", "ferrum-cli", "--test", "vllm_migration_compat", "serve_help_lists_vllm_compat_flags"],
        out / "serve-help-test.log",
        log,
        timeout=300,
    )
    checks["serve_help_test"] = True

    run(["cargo", "build", "--release", "-p", "ferrum-cli", "--bin", "ferrum"], out / "release-build.log", log, timeout=1200)
    help_proc = run([str(args.ferrum_bin), "serve", "--help"], out / "serve-help.txt", log, timeout=60)
    help_text = help_proc.stdout
    missing_flags = [flag for flag in REQUIRED_FLAGS if flag not in help_text]
    if missing_flags:
        raise RuntimeError(f"serve --help missing flags: {missing_flags}")
    checks["serve_help_flags"] = REQUIRED_FLAGS

    if not args.skip_ignored_rust_smoke:
        run(
            ["cargo", "test", "--release", "-p", "ferrum-cli", "--test", "vllm_migration_compat", "--", "--ignored", "--test-threads=1"],
            out / "rust-ignored-smoke.log",
            log,
            timeout=900,
        )
        checks["rust_ignored_smoke"] = True
    else:
        (out / "rust-ignored-smoke.log").write_text("skipped by --skip-ignored-rust-smoke\n")
        checks["rust_ignored_smoke"] = "skipped"

    checks["direct_openai_smoke"] = direct_smoke(args.ferrum_bin, args.model, args.model_name, out, log, args.port)
    (out / "smoke.json").write_text(json.dumps(checks["direct_openai_smoke"], ensure_ascii=False, indent=2) + "\n")

    (out / "gate.json").write_text(json.dumps({"status": "pass", "goal": "g1-vllm-migration", "checks": checks}, ensure_ascii=False, indent=2) + "\n")
    write_manifest(out, args, checks)
    print(f"G1 VLLM MIGRATION PASS: {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - command-line gate should report concise failure.
        print(f"G1 VLLM MIGRATION FAIL: {exc}", file=sys.stderr)
        raise
