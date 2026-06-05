#!/usr/bin/env python3
"""G1 vLLM migration compatibility release gate.

This gate is intentionally CPU/Metal-local by default. It verifies that the
vLLM-compatible serve flags are product-visible, reflected in the effective
runtime config, and compatible with OpenAI-style streaming clients.
"""
from __future__ import annotations

import argparse
import concurrent.futures
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

from g1_g4_manifest import required_manifest_fields, utc_now

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
    "--enable-prefix-cache",
    "--disable-prefix-cache",
]

REQUIRED_CONFIG = {
    "FERRUM_MAX_MODEL_LEN": "2048",
    "FERRUM_PAGED_MAX_SEQS": "1",
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


def run(
    cmd: list[str],
    out: Path,
    log: GateLog,
    *,
    timeout: int = 600,
    env: dict[str, str] | None = None,
    scan_bad_patterns: bool = False,
) -> subprocess.CompletedProcess[str]:
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
    if scan_bad_patterns:
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


def default_cargo_features() -> str:
    return "metal" if sys.platform == "darwin" else ""


def cargo_feature_list(features: str) -> list[str]:
    return [feature.strip() for feature in features.split(",") if feature.strip()]


def release_build_cmd(args: argparse.Namespace) -> list[str]:
    cmd = ["cargo", "build", "--release", "-p", "ferrum-cli", "--bin", "ferrum"]
    if args.cargo_features:
        cmd.extend(["--features", args.cargo_features])
    return cmd


def cargo_features_args(args: argparse.Namespace) -> list[str]:
    return ["--features", args.cargo_features] if args.cargo_features else []


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
    message = data["choices"][0]["message"]
    content = (
        message.get("content")
        or message.get("reasoning")
        or message.get("reasoning_content")
        or ""
    )
    assert_no_bad_patterns(label, body)
    if not content.strip():
        raise RuntimeError(f"empty chat content for {label}: {body[:500]}")
    if int(data.get("usage", {}).get("total_tokens", 0)) <= 0:
        raise RuntimeError(f"missing usage tokens for {label}: {body[:500]}")
    return content


def metric_value(metrics: str, name: str) -> float:
    for line in metrics.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0] == name:
            return float(parts[1])
    raise RuntimeError(f"missing metric {name}:\n{metrics}")


def require_openai_context_limit_error(label: str, status: int, body: str) -> dict[str, Any]:
    if status != 400:
        raise RuntimeError(f"{label} expected HTTP 400, got {status}: {body[:500]}")
    data = json.loads(body)
    error = data.get("error")
    if not isinstance(error, dict):
        raise RuntimeError(f"{label} did not return OpenAI-shaped error: {body[:500]}")
    if error.get("type") != "invalid_request_error":
        raise RuntimeError(f"{label} wrong error type: {error}")
    message = str(error.get("message") or "")
    lowered = message.lower()
    if not any(word in lowered for word in ["context", "model", "token", "length", "limited"]):
        raise RuntimeError(f"{label} error message does not explain context/model length: {message!r}")
    assert_no_bad_patterns(label, body)
    return {
        "status": status,
        "error_type": error.get("type"),
        "param": error.get("param"),
        "message": message,
    }


def check_context_limit_semantic(base: str, model_name: str, out: Path) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say one short word."}],
        "temperature": 0,
        "max_tokens": 4096,
    }
    status, body = request_json(base + "/v1/chat/completions", payload, timeout=30)
    (out / "semantic-max-model-len-400.json").write_text(body, errors="replace")
    result = require_openai_context_limit_error("semantic-max-model-len-400.json", status, body)
    result.update(
        {
            "configured_max_model_len": int(REQUIRED_CONFIG["FERRUM_MAX_MODEL_LEN"]),
            "requested_max_tokens": payload["max_tokens"],
            "observed_via": "OpenAI /v1/chat/completions HTTP 400 before generation",
        }
    )
    return result


def check_max_num_seqs_semantic(base: str, model_name: str, out: Path) -> dict[str, Any]:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Write five numbered words, one per line."}],
        "temperature": 0,
        "max_tokens": 256,
    }

    def post_one(index: int) -> tuple[int, str]:
        indexed = {
            **payload,
            "messages": [
                {
                    "role": "user",
                    "content": f"Request {index}: Write five numbered words, one per line.",
                }
            ],
        }
        return request_json(base + "/v1/chat/completions", indexed, timeout=300)

    samples: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(post_one, i) for i in range(2)]
        deadline = time.time() + 300
        while time.time() < deadline and not all(f.done() for f in futures):
            try:
                status, body = request_json(base + "/health", None, timeout=2)
                if status == 200:
                    health = json.loads(body)
                    samples.append(
                        {
                            "active": int(health["engine"]["active_requests"]),
                            "queued": int(health["engine"]["queued_requests"]),
                        }
                    )
            except Exception:
                pass
            time.sleep(0.05)
        responses = [future.result(timeout=1) for future in futures]

    response_summaries = []
    for idx, (status, body) in enumerate(responses):
        (out / f"semantic-max-num-seqs-response-{idx}.json").write_text(body, errors="replace")
        if status != 200:
            raise RuntimeError(f"max-num-seqs request {idx} failed status={status}: {body[:500]}")
        content = require_nonempty_chat(f"semantic-max-num-seqs-response-{idx}.json", body)
        response_summaries.append({"status": status, "content_len": len(content)})

    max_active = max((sample["active"] for sample in samples), default=0)
    max_queued = max((sample["queued"] for sample in samples), default=0)
    result = {
        "configured_max_num_seqs": int(REQUIRED_CONFIG["FERRUM_PAGED_MAX_SEQS"]),
        "max_observed_active_requests": max_active,
        "max_observed_queued_requests": max_queued,
        "health_samples": samples[:200],
        "responses": response_summaries,
        "observed_via": "two concurrent OpenAI chat requests plus /health active/queued polling",
    }
    (out / "semantic-max-num-seqs.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if max_active > int(REQUIRED_CONFIG["FERRUM_PAGED_MAX_SEQS"]):
        raise RuntimeError(f"max active requests {max_active} exceeded configured max-num-seqs")
    if max_queued < 1:
        raise RuntimeError(f"concurrency probe never observed queued request: {result}")
    return result


def chat_once(base: str, model_name: str, content: str, artifact: Path) -> str:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 128,
    }
    status, body = request_json(base + "/v1/chat/completions", payload, timeout=120)
    artifact.write_text(body, errors="replace")
    if status != 200:
        raise RuntimeError(f"{artifact.name} chat failed status={status}: {body[:500]}")
    return require_nonempty_chat(artifact.name, body)


def run_prefix_alias_probe(
    bin_path: Path,
    model: str,
    model_name: str,
    out: Path,
    log: GateLog,
    *,
    label: str,
    flag: str,
    expect_enabled: bool,
) -> dict[str, Any]:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    serve_log = out / f"serve-prefix-alias-{label}.log"
    cmd = [
        str(bin_path),
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        flag,
        "--session-cache",
        "off",
    ]
    log.write("START " + " ".join(cmd))
    with serve_log.open("wb") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "NO_COLOR": "1"},
        )
    try:
        wait_health(base, 180, log)
        prompt = (
            "Ferrum prefix-cache verification prompt. The shared prefix is intentionally long "
            "and stable so it crosses at least two paged-KV blocks before the requested answer. "
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron "
            "pi rho sigma tau upsilon phi chi psi omega. Reply with exactly: ferrum-cache-ok"
        )
        first = chat_once(base, model_name, prompt, out / f"semantic-prefix-alias-{label}-chat-1.json")
        second = chat_once(base, model_name, prompt, out / f"semantic-prefix-alias-{label}-chat-2.json")

        health_status, health_body = request_json(base + "/health", None, timeout=30)
        if health_status != 200:
            raise RuntimeError(f"{label} /health failed status={health_status}: {health_body[:500]}")
        (out / f"semantic-prefix-alias-{label}-health.json").write_text(health_body, errors="replace")
        health = json.loads(health_body)
        prefix = health["cache"]["prefix_cache"]

        metrics_status, metrics_body = request_json(base + "/metrics", None, timeout=30)
        if metrics_status != 200:
            raise RuntimeError(f"{label} /metrics failed status={metrics_status}: {metrics_body[:500]}")
        (out / f"semantic-prefix-alias-{label}-metrics.txt").write_text(metrics_body, errors="replace")

        hits_metric = metric_value(metrics_body, "ferrum_prefix_cache_hits_total")
        saved_metric = metric_value(metrics_body, "ferrum_prefix_cache_saved_prefill_tokens_total")
        enabled = bool(prefix["enabled"])
        hits = float(prefix["hits"])
        saved = float(prefix["saved_prefill_tokens"])
        result = {
            "flag": flag,
            "expected_enabled": expect_enabled,
            "health_enabled": enabled,
            "health_hits": hits,
            "health_saved_prefill_tokens": saved,
            "metrics_hits": hits_metric,
            "metrics_saved_prefill_tokens": saved_metric,
            "cache_position": prefix.get("position"),
            "first_content_len": len(first),
            "second_content_len": len(second),
            "observed_via": "two identical OpenAI chat requests plus /health and /metrics",
        }
        if enabled != expect_enabled:
            raise RuntimeError(f"{label} enabled={enabled}, expected {expect_enabled}: {result}")
        if expect_enabled:
            if hits <= 0 or saved <= 0 or hits_metric <= 0 or saved_metric <= 0:
                raise RuntimeError(f"{label} expected prefix hits and saved tokens: {result}")
        else:
            if hits != 0 or saved != 0 or hits_metric != 0 or saved_metric != 0:
                raise RuntimeError(f"{label} expected zero disabled prefix metrics: {result}")
        return result
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        serve_text = serve_log.read_text(errors="replace") if serve_log.exists() else ""
        assert_no_bad_patterns(serve_log.name, serve_text)


def check_prefix_alias_runtime_semantics(
    bin_path: Path,
    model: str,
    model_name: str,
    out: Path,
    log: GateLog,
) -> dict[str, Any]:
    probes = {
        "enable_vllm": ("--enable-prefix-caching", True),
        "enable_product": ("--enable-prefix-cache", True),
        "disable_vllm": ("--no-enable-prefix-caching", False),
        "disable_product": ("--disable-prefix-cache", False),
    }
    results = {
        label: run_prefix_alias_probe(
            bin_path,
            model,
            model_name,
            out,
            log,
            label=label,
            flag=flag,
            expect_enabled=expect_enabled,
        )
        for label, (flag, expect_enabled) in probes.items()
    }
    enabled_hits = [results["enable_vllm"]["health_hits"], results["enable_product"]["health_hits"]]
    enabled_saved = [
        results["enable_vllm"]["health_saved_prefill_tokens"],
        results["enable_product"]["health_saved_prefill_tokens"],
    ]
    disabled_hits = [results["disable_vllm"]["health_hits"], results["disable_product"]["health_hits"]]
    summary = {
        "passed": True,
        "enable_aliases": ["--enable-prefix-caching", "--enable-prefix-cache"],
        "disable_aliases": ["--no-enable-prefix-caching", "--disable-prefix-cache"],
        "enabled_hits": enabled_hits,
        "enabled_saved_prefill_tokens": enabled_saved,
        "disabled_hits": disabled_hits,
        "results": results,
    }
    (out / "semantic-prefix-aliases-runtime.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return summary


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
            "max_tokens": 128,
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

        context_limit = check_context_limit_semantic(base, model_name, out)
        max_num_seqs = check_max_num_seqs_semantic(base, model_name, out)

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
            "semantic_tests": {
                "max_model_len": context_limit,
                "max_num_seqs": max_num_seqs,
            },
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


def write_manifest(
    out: Path,
    args: argparse.Namespace,
    checks: dict[str, Any],
    started_at_utc: str,
) -> None:
    manifest = {
        **required_manifest_fields(
            repo=repo_root(),
            goal="G1",
            name="vllm-migration",
            models=[args.model],
            commands=[
                "cargo test --workspace --all-targets",
                "cargo test -p ferrum-cli --test vllm_migration_compat serve_help_lists_vllm_compat_flags",
                "cargo test -p ferrum-types --test config_tests engine_config_applies_runtime_snapshot",
                "cargo test -p ferrum-scheduler max_",
                "cargo test -p ferrum-cli prefix_cache_vllm_and_product_aliases_resolve_identically",
                "cargo build --release -p ferrum-cli --bin ferrum"
                + (f" --features {args.cargo_features}" if args.cargo_features else ""),
                "ferrum serve --help",
                "cargo test --release -p ferrum-cli --test vllm_migration_compat -- --ignored --test-threads=1",
                "direct OpenAI-compatible smoke plus max-model-len and max-num-seqs semantic probes",
                "direct prefix-cache alias runtime probes with /health and /metrics",
            ],
            started_at_utc=started_at_utc,
            binary_path=args.ferrum_bin,
            features=cargo_feature_list(args.cargo_features),
        ),
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
        "- `--max-model-len` returned OpenAI-shaped HTTP 400 when prompt + max_tokens exceeded the configured context limit.\n"
        "- `--max-num-seqs` was tested with concurrent product requests and `/health` active/queued observations.\n"
        "- `--max-num-batched-tokens` was tested at scheduler batch-plan level with prompt-token admission evidence.\n"
        "- Prefix-cache vLLM/product enable aliases both produced runtime cache hits and saved prefill tokens.\n"
        "- Prefix-cache vLLM/product disable aliases both kept runtime cache hits at zero.\n"
        "- `/v1/models`, non-stream chat, stream chat, and usage stream smoke passed.\n"
        "- Streaming emitted exactly one `[DONE]`; usage stream emitted exactly one usage chunk before `[DONE]`.\n"
        "- Server log was scanned for forbidden release-blocker patterns.\n",
    )


def main() -> int:
    started_at_utc = utc_now()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="artifact directory")
    parser.add_argument("--model", default="qwen3:0.6b", help="local Ferrum model path or alias")
    parser.add_argument("--model-name", default=None, help="model field sent to OpenAI API; defaults to --model")
    parser.add_argument("--ferrum-bin", type=Path, default=repo_root() / "target" / "release" / "ferrum")
    parser.add_argument("--cargo-features", default=default_cargo_features())
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
    run(
        ["cargo", "test", "-p", "ferrum-types", "--test", "config_tests", "engine_config_applies_runtime_snapshot"],
        out / "semantic-engine-config.log",
        log,
        timeout=300,
    )
    checks["semantic_engine_config"] = {
        "passed": True,
        "covers": ["FERRUM_PAGED_MAX_SEQS -> scheduler.max_running_requests"],
    }
    run(
        ["cargo", "test", "-p", "ferrum-scheduler", "max_"],
        out / "semantic-scheduler.log",
        log,
        timeout=300,
    )
    checks["semantic_scheduler"] = {
        "passed": True,
        "covers": [
            "--max-num-seqs active admission cap",
            "--max-num-batched-tokens iteration batch token budget",
        ],
    }
    run(
        ["cargo", "test", "-p", "ferrum-cli", "prefix_cache_vllm_and_product_aliases_resolve_identically"],
        out / "semantic-prefix-aliases.log",
        log,
        timeout=300,
    )
    checks["semantic_prefix_aliases"] = {
        "passed": True,
        "covers": [
            "--enable-prefix-caching == --enable-prefix-cache",
            "--no-enable-prefix-caching == --disable-prefix-cache",
        ],
    }

    run(release_build_cmd(args), out / "release-build.log", log, timeout=1200)
    help_proc = run([str(args.ferrum_bin), "serve", "--help"], out / "serve-help.txt", log, timeout=60)
    help_text = help_proc.stdout
    missing_flags = [flag for flag in REQUIRED_FLAGS if flag not in help_text]
    if missing_flags:
        raise RuntimeError(f"serve --help missing flags: {missing_flags}")
    checks["serve_help_flags"] = REQUIRED_FLAGS

    if not args.skip_ignored_rust_smoke:
        run(
            [
                "cargo",
                "test",
                "--release",
                "-p",
                "ferrum-cli",
                *cargo_features_args(args),
                "--test",
                "vllm_migration_compat",
                "--",
                "--ignored",
                "--test-threads=1",
            ],
            out / "rust-ignored-smoke.log",
            log,
            timeout=900,
            env={"FERRUM_G1_SMOKE_MODEL": args.model},
        )
        checks["rust_ignored_smoke"] = True
    else:
        (out / "rust-ignored-smoke.log").write_text("skipped by --skip-ignored-rust-smoke\n")
        checks["rust_ignored_smoke"] = "skipped"

    checks["direct_openai_smoke"] = direct_smoke(args.ferrum_bin, args.model, args.model_name, out, log, args.port)
    checks["prefix_alias_runtime"] = check_prefix_alias_runtime_semantics(
        args.ferrum_bin,
        args.model,
        args.model_name,
        out,
        log,
    )
    (out / "smoke.json").write_text(json.dumps(checks["direct_openai_smoke"], ensure_ascii=False, indent=2) + "\n")
    semantic_tests = {
        "max_model_len": checks["direct_openai_smoke"]["semantic_tests"]["max_model_len"],
        "max_num_seqs": checks["direct_openai_smoke"]["semantic_tests"]["max_num_seqs"],
        "max_num_batched_tokens": {
            "passed": checks["semantic_scheduler"]["passed"],
            "observed_via": "ferrum-scheduler unit test max_batched_tokens_limits_prefill_admission_by_prompt_tokens",
            "configured_value": int(REQUIRED_CONFIG["FERRUM_MAX_BATCHED_TOKENS"]),
            "artifact": "semantic-scheduler.log",
        },
        "prefix_cache_aliases": checks["prefix_alias_runtime"],
    }
    (out / "semantic-tests.json").write_text(json.dumps(semantic_tests, ensure_ascii=False, indent=2) + "\n")

    (out / "gate.json").write_text(json.dumps({"status": "pass", "goal": "g1-vllm-migration", "checks": checks}, ensure_ascii=False, indent=2) + "\n")
    write_manifest(out, args, checks, started_at_utc)
    print(f"G1 VLLM MIGRATION PASS: {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - command-line gate should report concise failure.
        print(f"G1 VLLM MIGRATION FAIL: {exc}", file=sys.stderr)
        raise
