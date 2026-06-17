#!/usr/bin/env python3
"""W3-S2 Qwen3.5 product-entry smoke artifact.

This is a small product-path gate for the explicit CPU/FP32 Qwen3.5 reference
executor. It creates a tiny local Qwen3.5 safetensors model, runs real
`ferrum run`, starts real `ferrum serve`, exercises non-streaming and streaming
OpenAI chat completions, and writes an artifact manifest consumable by the
release-grade W3 goal gate as `w3_s2_whole_model_product_path`.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL_DOC = "docs/goals/model-coverage-2026-06-12/RELEASE_GRADE_GOAL.md"
PASS_LINE_PREFIX = "W3 QWEN35 PRODUCT SMOKE PASS"
MANIFEST_NAME = "w3_s2_whole_model_product_path.json"
STARTUP_TIMEOUT_SECONDS = 60.0
REQUEST_TIMEOUT_SECONDS = 30.0


class SmokeError(Exception):
    pass


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def command_line() -> list[str]:
    return [sys.executable, *sys.argv]


def run_command(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def ferrum_command(args: list[str], ferrum_bin: Path | None) -> list[str]:
    if ferrum_bin is not None:
        return [str(ferrum_bin), *args]
    debug_bin = REPO_ROOT / "target/debug/ferrum"
    if debug_bin.is_file():
        return [str(debug_bin), *args]
    return ["cargo", "run", "-p", "ferrum-cli", "--bin", "ferrum", "--", *args]


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    try:
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def write_qwen35_reference_config(model_dir: Path) -> None:
    write_json(
        model_dir / "config.json",
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "vocab_size": 3,
            "max_position_embeddings": 16,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 2,
                "intermediate_size": 2,
                "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "full_attention"],
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 1,
                "linear_key_head_dim": 1,
                "linear_value_head_dim": 1,
                "linear_conv_kernel_dim": 1,
                "head_dim": 2,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "tie_word_embeddings": False,
            },
        },
    )


def write_qwen35_reference_tokenizer(model_dir: Path) -> None:
    write_json(
        model_dir / "tokenizer.json",
        {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [],
            "normalizer": None,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": None,
            "decoder": None,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<unk>": 0,
                    "alpha": 1,
                    "beta": 2,
                },
                "unk_token": "<unk>",
            },
        },
    )


def qwen35_reference_tensors() -> dict[str, list[float]]:
    return {
        "model.embed_tokens.weight": [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "model.norm.weight": [0.0, 0.0],
        "model.lm_head.weight": [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "model.layers.0.input_layernorm.weight": [0.0, 0.0],
        "model.layers.0.post_attention_layernorm.weight": [0.0, 0.0],
        "model.layers.0.linear_attn.in_proj_qkv.weight": [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "model.layers.0.linear_attn.in_proj_z.weight": [1.0, -1.0],
        "model.layers.0.linear_attn.in_proj_b.weight": [0.5, 0.25],
        "model.layers.0.linear_attn.in_proj_a.weight": [-0.25, 0.75],
        "model.layers.0.linear_attn.conv1d.weight": [1.0, 1.0, 1.0],
        "model.layers.0.linear_attn.A_log": [0.0],
        "model.layers.0.linear_attn.dt_bias": [0.0],
        "model.layers.0.linear_attn.norm.weight": [1.0],
        "model.layers.0.linear_attn.out_proj.weight": [1.0, -0.5],
        "model.layers.0.mlp.gate_proj.weight": [0.2, 0.1, -0.1, 0.3],
        "model.layers.0.mlp.up_proj.weight": [0.4, -0.2, 0.3, 0.5],
        "model.layers.0.mlp.down_proj.weight": [1.0, 0.0, 0.0, 1.0],
        "model.layers.1.input_layernorm.weight": [0.0, 0.0],
        "model.layers.1.post_attention_layernorm.weight": [0.0, 0.0],
        "model.layers.1.self_attn.q_proj.weight": [1.0, 0.0, 0.0, 1.0],
        "model.layers.1.self_attn.k_proj.weight": [0.5, 0.0, 0.0, 0.5],
        "model.layers.1.self_attn.v_proj.weight": [1.0, 1.0, -0.5, 0.5],
        "model.layers.1.self_attn.o_proj.weight": [1.0, 0.0, 0.0, 1.0],
        "model.layers.1.self_attn.q_norm.weight": [1.0, 1.0],
        "model.layers.1.self_attn.k_norm.weight": [1.0, 1.0],
        "model.layers.1.mlp.gate_proj.weight": [-0.2, 0.2, 0.1, 0.3],
        "model.layers.1.mlp.up_proj.weight": [0.25, 0.5, -0.3, 0.4],
        "model.layers.1.mlp.down_proj.weight": [0.5, 0.25, -0.2, 0.75],
    }


def write_safetensors_f32(path: Path, tensors: dict[str, list[float]]) -> None:
    header: dict[str, Any] = {}
    data = bytearray()
    for name, values in tensors.items():
        start = len(data)
        for value in values:
            data.extend(struct.pack("<f", float(value)))
        end = len(data)
        header[name] = {
            "dtype": "F32",
            "shape": [len(values)],
            "data_offsets": [start, end],
        }
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    padding = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * padding
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(data))


def write_model_dir(root: Path) -> Path:
    model_dir = root / "qwen35-reference-model"
    model_dir.mkdir(parents=True, exist_ok=True)
    write_qwen35_reference_config(model_dir)
    write_qwen35_reference_tokenizer(model_dir)
    write_safetensors_f32(model_dir / "model.safetensors", qwen35_reference_tensors())
    return model_dir


def product_env(out_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    env["HF_HOME"] = str(out_dir / "hf-cache")
    return env


def validate_run_stdout(stdout: str) -> dict[str, Any]:
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "assistant":
            continue
        if event.get("finish_reason") != "length":
            raise SmokeError(f"ferrum run finish_reason mismatch: {event}")
        if event.get("n_tokens") != 2:
            raise SmokeError(f"ferrum run n_tokens mismatch: {event}")
        content = event.get("content")
        if not isinstance(content, str) or not content.strip():
            raise SmokeError(f"ferrum run content empty: {event}")
        return event
    raise SmokeError(f"missing assistant JSONL event in ferrum run stdout: {stdout}")


def run_ferrum_run(out_dir: Path, model_dir: Path, ferrum_bin: Path | None) -> dict[str, Any]:
    cmd = ferrum_command(
        [
            "run",
            str(model_dir),
            "--backend",
            "cpu",
            "--qwen35-reference",
            "--output-format",
            "jsonl",
            "--temperature",
            "0",
            "--max-tokens",
            "2",
            "--prompt",
            "hello",
        ],
        ferrum_bin,
    )
    proc = run_command(cmd, cwd=out_dir, env=product_env(out_dir))
    write_json(out_dir / "run_command.json", {"command_line": cmd, "returncode": proc.returncode})
    write_text(out_dir / "run_stdout.jsonl", proc.stdout)
    write_text(out_dir / "run_stderr.txt", proc.stderr)
    if proc.returncode != 0:
        raise SmokeError(f"ferrum run failed with rc={proc.returncode}: {proc.stderr}")
    assistant = validate_run_stdout(proc.stdout)
    return {
        "status": "pass",
        "command_line": cmd,
        "stdout": artifact_ref(out_dir / "run_stdout.jsonl", out_dir),
        "stderr": artifact_ref(out_dir / "run_stderr.txt", out_dir),
        "assistant_event": assistant,
    }


def http_get_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            return 200 <= int(response.status) < 300
    except Exception:
        return False


def http_post_json(url: str, payload: dict[str, Any]) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return int(response.status), response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")


def parse_sse(body: str) -> tuple[list[dict[str, Any]], int]:
    chunks: list[dict[str, Any]] = []
    done_count = 0
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :].strip()
        if data == "[DONE]":
            done_count += 1
        elif data:
            chunks.append(json.loads(data))
    return chunks, done_count


def validate_nonstream(body: str, model_id: str) -> dict[str, Any]:
    parsed = json.loads(body)
    if parsed.get("model") != model_id:
        raise SmokeError(f"serve non-stream model mismatch: {parsed}")
    choice = parsed.get("choices", [{}])[0]
    if choice.get("finish_reason") != "length":
        raise SmokeError(f"serve non-stream finish_reason mismatch: {parsed}")
    content = choice.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise SmokeError(f"serve non-stream content empty: {parsed}")
    return parsed


def validate_stream(body: str) -> dict[str, Any]:
    chunks, done_count = parse_sse(body)
    if done_count != 1:
        raise SmokeError(f"serve stream expected one [DONE], got {done_count}: {body}")
    has_delta = any(
        isinstance(chunk.get("choices"), list)
        and chunk["choices"]
        and isinstance(chunk["choices"][0], dict)
        and isinstance(chunk["choices"][0].get("delta"), dict)
        and isinstance(chunk["choices"][0]["delta"].get("content"), str)
        and chunk["choices"][0]["delta"]["content"]
        for chunk in chunks
    )
    if not has_delta:
        raise SmokeError(f"serve stream missing content delta: {body}")
    has_usage = any(chunk.get("usage") is not None for chunk in chunks)
    if not has_usage:
        raise SmokeError(f"serve stream missing usage chunk: {body}")
    return {"chunk_count": len(chunks), "done_count": done_count, "has_usage": has_usage}


def run_ferrum_serve(out_dir: Path, model_dir: Path, ferrum_bin: Path | None) -> dict[str, Any]:
    port = free_port()
    model_id = model_dir.name
    cmd = ferrum_command(
        [
            "serve",
            str(model_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--backend",
            "cpu",
            "--qwen35-reference",
        ],
        ferrum_bin,
    )
    log_path = out_dir / "serve.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=out_dir,
            env=product_env(out_dir),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            health_url = f"http://127.0.0.1:{port}/health"
            deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise SmokeError(f"ferrum serve exited early with rc={proc.returncode}")
                if http_get_ok(health_url):
                    break
                time.sleep(0.25)
            else:
                raise SmokeError(f"ferrum serve did not become healthy: {log_path}")

            chat_url = f"http://127.0.0.1:{port}/v1/chat/completions"
            nonstream_payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2,
                "temperature": 0.0,
                "stream": False,
            }
            status, nonstream_body = http_post_json(chat_url, nonstream_payload)
            write_text(out_dir / "serve_nonstream.json", nonstream_body)
            if status < 200 or status >= 300:
                raise SmokeError(f"serve non-stream HTTP {status}: {nonstream_body}")
            nonstream = validate_nonstream(nonstream_body, model_id)

            stream_payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            stream_status, stream_body = http_post_json(chat_url, stream_payload)
            write_text(out_dir / "serve_stream.sse", stream_body)
            if stream_status < 200 or stream_status >= 300:
                raise SmokeError(f"serve stream HTTP {stream_status}: {stream_body}")
            stream = validate_stream(stream_body)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)

    write_json(out_dir / "serve_command.json", {"command_line": cmd, "port": port})
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    for bad in ["panicked", "KV cache overflow", "stream error"]:
        if bad in log_text:
            raise SmokeError(f"serve log contains {bad}: {log_path}")
    return {
        "status": "pass",
        "command_line": cmd,
        "log": artifact_ref(log_path, out_dir),
        "nonstream": {
            "artifact": artifact_ref(out_dir / "serve_nonstream.json", out_dir),
            "finish_reason": nonstream["choices"][0]["finish_reason"],
            "content_len": len(nonstream["choices"][0]["message"]["content"]),
        },
        "stream": {
            "artifact": artifact_ref(out_dir / "serve_stream.sse", out_dir),
            **stream,
        },
    }


def run_smoke(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = write_model_dir(out_dir)
    ferrum_bin = Path(args.ferrum_bin).resolve() if args.ferrum_bin else None
    run_result = run_ferrum_run(out_dir, model_dir, ferrum_bin)
    serve_result = run_ferrum_serve(out_dir, model_dir, ferrum_bin)
    pass_line = f"{PASS_LINE_PREFIX}: {out_dir}"
    write_json(
        out_dir / MANIFEST_NAME,
        {
            "schema_version": 1,
            "status": "pass",
            "lane": "w3_s2_whole_model_product_path",
            "goal_doc": GOAL_DOC,
            "pass_line": pass_line,
            "created_at": iso_now(),
            "command_line": command_line(),
            "git": git_summary(),
            "model_id": "qwen35-reference-model",
            "architecture": "qwen3_5",
            "backend": "cpu",
            "quantization": "fp32-reference",
            "runtime_surface": "typed_cli",
            "hidden_env": [],
            "model_dir": artifact_ref(model_dir, out_dir),
            "product_entrypoints": {
                "ferrum_run": run_result,
                "ferrum_serve": serve_result,
            },
            "limitations": [
                "toy CPU/FP32 reference product smoke only",
                "not real-model L0-L5 correctness",
                "not CUDA/Metal performance evidence",
            ],
        },
    )
    print(pass_line)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="artifact output directory")
    parser.add_argument(
        "--ferrum-bin",
        type=Path,
        help="path to an existing ferrum binary; defaults to target/debug/ferrum or cargo run",
    )
    args = parser.parse_args()
    try:
        return run_smoke(args)
    except SmokeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
