#!/usr/bin/env python3
"""Collect a provenance-bound vLLM raw-logits reference for one C13 request."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
import platform
import shlex
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
COLLECTOR_ID = "ferrum.runtime-vnext.c13-vllm-raw-logits.v1"
TOKEN_SPAN_DOMAIN = b"ferrum.runtime-vnext.token-span-work.v3\0"
PASS_LINE = "FERRUM C13 VLLM RAW LOGITS REFERENCE PASS"
REPO_ROOT = Path(__file__).resolve().parents[2]
C13_REQUEST = (
    REPO_ROOT / "scripts/release/configs/runtime_vnext_c13_022_reference.json"
)
C13_REQUEST_SHA256 = "fa92ee4502f97a2d020d9c3d1123be53d2e0825410d6fe0321c19654eddb419e"


class ReferenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReferenceError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


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
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_json_bytes(value))


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReferenceError(f"cannot read JSON object {path}: {error}") from error
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def git_capture(source: Path) -> dict[str, Any]:
    require((source / ".git").exists(), f"vLLM source is not a git worktree: {source}")

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(source), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        sha = run("rev-parse", "HEAD")
        status = run("status", "--short")
        remote = run("remote", "get-url", "origin")
    except subprocess.CalledProcessError as error:
        raise ReferenceError(f"cannot inspect vLLM git source: {error}") from error
    require(len(sha) == 40, "vLLM git SHA must be a full 40-character SHA")
    return {
        "source_path": str(source.resolve()),
        "git_sha": sha,
        "git_status": status,
        "dirty": bool(status),
        "origin": remote,
    }


def token_span_fingerprint(token_ids: list[int]) -> str:
    require(bool(token_ids), "prompt token IDs must not be empty")
    digest = hashlib.sha256()
    count = len(token_ids)
    digest.update(TOKEN_SPAN_DOMAIN)
    digest.update(struct.pack("<Q", count))
    digest.update(struct.pack("<Q", count))
    digest.update(struct.pack("<Q", 0))
    digest.update(struct.pack("<Q", count))
    for token_id in token_ids:
        require(
            isinstance(token_id, int) and 0 <= token_id <= 0xFFFFFFFF,
            f"invalid prompt token ID: {token_id!r}",
        )
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def snapshot_revision(path: Path, expected: str, label: str) -> None:
    require(path.is_dir(), f"{label} path does not exist: {path}")
    require(len(expected) == 40, f"{label} revision must be a full 40-character SHA")
    require(
        path.resolve().name == expected,
        f"{label} snapshot directory {path.resolve().name!r} does not match revision {expected}",
    )


def top_logits(values: list[float], count: int = 20) -> list[dict[str, Any]]:
    ranked = sorted(range(len(values)), key=lambda token_id: (-values[token_id], token_id))
    return [
        {"token_id": token_id, "logit": values[token_id], "rank": rank}
        for rank, token_id in enumerate(ranked[:count], start=1)
    ]


def command_line() -> str:
    return " ".join(shlex.quote(part) for part in sys.argv)


def collect(args: argparse.Namespace) -> Path:
    request_path = args.request.resolve()
    model_path = args.model.resolve()
    tokenizer_path = args.tokenizer.resolve()
    vllm_source = args.vllm_source.resolve()
    out_dir = args.out.resolve()

    snapshot_revision(model_path, args.model_revision, "model")
    snapshot_revision(tokenizer_path, args.tokenizer_revision, "tokenizer")
    require(
        sha256_file(request_path) == C13_REQUEST_SHA256,
        "request does not match the checked-in canonical c13-022 input",
    )
    request = load_object(request_path)
    messages = request.get("messages")
    require(isinstance(messages, list) and messages, "request.messages must be non-empty")
    chat_template_kwargs = request.get("chat_template_kwargs", {})
    require(
        isinstance(chat_template_kwargs, dict),
        "request.chat_template_kwargs must be an object",
    )
    tools = request.get("tools")
    require(tools is None or isinstance(tools, list), "request.tools must be a list")
    metadata = request.get("metadata", {})
    require(isinstance(metadata, dict), "request.metadata must be an object")

    vllm_git = git_capture(vllm_source)
    require(
        not vllm_git["dirty"],
        "vLLM reference source must be clean; dirty state was:\n"
        + str(vllm_git["git_status"]),
    )

    # Imports remain inside the paid-GPU collection path so --self-test works
    # on development machines without torch or vLLM.
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise ReferenceError(f"vLLM collection dependencies are unavailable: {error}") from error
    package_path = Path(vllm.__file__).resolve()
    require(
        package_path.is_relative_to(vllm_source),
        f"imported vLLM package {package_path} is not from {vllm_source}",
    )

    sampling = {
        "max_tokens": 1,
        "temperature": float(request.get("temperature", 1.0)),
        "top_p": float(request.get("top_p", 1.0)),
        "top_k": int(request.get("top_k", 0)),
        "min_p": float(request.get("min_p", 0.0)),
        "presence_penalty": float(request.get("presence_penalty", 0.0)),
        "frequency_penalty": float(request.get("frequency_penalty", 0.0)),
        "repetition_penalty": float(request.get("repetition_penalty", 1.0)),
        "seed": int(request["seed"]) if request.get("seed") is not None else None,
        "stop": request.get("stop", []),
        "logprobs": -1,
    }
    params = SamplingParams(**sampling)
    llm_kwargs = {
        "model": str(model_path),
        "tokenizer": str(tokenizer_path),
        "dtype": args.dtype,
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "max_logprobs": -1,
        "logprobs_mode": "raw_logits",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb,
        "max_model_len": args.max_model_len,
        "max_num_seqs": 1,
        "max_num_batched_tokens": args.max_model_len,
        "seed": args.engine_seed,
        "language_model_only": True,
    }

    llm = LLM(**llm_kwargs)
    try:
        outputs = llm.chat(
            messages,
            sampling_params=params,
            use_tqdm=False,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        require(len(outputs) == 1, f"expected one vLLM output, observed {len(outputs)}")
        output = outputs[0]
        prompt_token_ids = list(output.prompt_token_ids or [])
        fingerprint = token_span_fingerprint(prompt_token_ids)
        require(
            len(prompt_token_ids) == args.expected_prompt_tokens,
            f"prompt token count mismatch: expected {args.expected_prompt_tokens}, "
            f"observed {len(prompt_token_ids)}",
        )
        require(
            fingerprint == args.expected_token_span_fingerprint,
            "prompt token fingerprint mismatch: "
            f"expected {args.expected_token_span_fingerprint}, observed {fingerprint}",
        )
        require(len(output.outputs) == 1, "expected one completion")
        completion = output.outputs[0]
        require(completion.logprobs is not None, "vLLM did not return sample logprobs")
        require(len(completion.logprobs) == 1, "expected one raw-logits position")
        position = completion.logprobs[0]
        require(isinstance(position, dict) and position, "raw-logits position is empty")

        vocab_size = len(position)
        token_ids = sorted(position)
        require(
            token_ids == list(range(vocab_size)),
            "raw-logits token IDs must cover the contiguous full vocabulary",
        )
        values = [float(position[token_id].logprob) for token_id in token_ids]
        require(
            all(value == value and abs(value) != float("inf") for value in values),
            "raw logits contain NaN or infinity",
        )

        raw_values = array.array("f", values)
        if sys.byteorder != "little":
            raw_values.byteswap()
        raw_path = out_dir / "raw-logits.f32le"
        atomic_write(raw_path, raw_values.tobytes())
        raw_sha256 = sha256_file(raw_path)

        generated_token_ids = list(completion.token_ids)
        require(len(generated_token_ids) == 1, "expected exactly one generated token")
        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        }
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            cuda.update(
                {
                    "device_name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [properties.major, properties.minor],
                }
            )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "collector_id": COLLECTOR_ID,
            "status": "pass",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "command": command_line(),
            "host": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python": sys.version,
                "cuda": cuda,
            },
            "vllm": {
                **vllm_git,
                "package_version": getattr(vllm, "__version__", "unknown"),
            },
            "model": {
                "model_id": args.model_id,
                "model_path": str(model_path),
                "model_revision": args.model_revision,
                "tokenizer_path": str(tokenizer_path),
                "tokenizer_revision": args.tokenizer_revision,
                "dtype": args.dtype,
            },
            "request": {
                "path": str(request_path),
                "sha256": sha256_file(request_path),
                "case_id": metadata.get("g00_case_id"),
                "chat_template_kwargs": chat_template_kwargs,
            },
            "engine": llm_kwargs,
            "sampling": sampling,
            "prompt": {
                "token_count": len(prompt_token_ids),
                "token_ids": prompt_token_ids,
                "token_ids_sha256": sha256_bytes(
                    b"".join(struct.pack("<I", token_id) for token_id in prompt_token_ids)
                ),
                "token_span_fingerprint": fingerprint,
            },
            "output": {
                "generated_token_ids": generated_token_ids,
                "generated_text": completion.text,
                "raw_logits": {
                    "file": raw_path.name,
                    "element_type": "f32le",
                    "element_count": vocab_size,
                    "bytes": raw_path.stat().st_size,
                    "sha256": raw_sha256,
                },
                "top20": top_logits(values),
            },
        }
        manifest_path = out_dir / "reference.json"
        atomic_write_json(manifest_path, manifest)
        return manifest_path
    finally:
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def self_test() -> None:
    require(C13_REQUEST.is_file(), "checked-in canonical c13-022 request is missing")
    require(
        sha256_file(C13_REQUEST) == C13_REQUEST_SHA256,
        "checked-in canonical c13-022 request SHA256 drifted",
    )
    expected = "4bb2e94bd2076d0f6f2663ccdbf7e63abc4bc809462d5f940f1132cfdeccddeb"
    require(
        token_span_fingerprint([1, 2, 3]) == expected,
        "token-span fingerprint version vector drifted",
    )
    require(
        token_span_fingerprint([1, 2, 4]) != expected,
        "token-span fingerprint did not bind token identity",
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "nested" / "manifest.json"
        atomic_write_json(path, {"b": 2, "a": 1})
        require(path.read_bytes() == b'{"a":1,"b":2}\n', "canonical JSON write drifted")
    print(f"{PASS_LINE}: self-test")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--self-test", action="store_true")
    result.add_argument("--model", type=Path)
    result.add_argument("--tokenizer", type=Path)
    result.add_argument("--model-id")
    result.add_argument("--model-revision")
    result.add_argument("--tokenizer-revision")
    result.add_argument("--request", type=Path)
    result.add_argument("--vllm-source", type=Path)
    result.add_argument("--out", type=Path)
    result.add_argument("--expected-prompt-tokens", type=int)
    result.add_argument("--expected-token-span-fingerprint")
    result.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    result.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    result.add_argument("--cpu-offload-gb", type=float, default=0.0)
    result.add_argument("--max-model-len", type=int, default=256)
    result.add_argument("--engine-seed", type=int, default=9271)
    return result


def validate_args(args: argparse.Namespace) -> None:
    required = (
        "model",
        "tokenizer",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "request",
        "vllm_source",
        "out",
        "expected_prompt_tokens",
        "expected_token_span_fingerprint",
    )
    missing = [name for name in required if getattr(args, name) is None]
    require(not missing, "missing required arguments: " + ", ".join(missing))
    require(args.expected_prompt_tokens > 0, "--expected-prompt-tokens must be positive")
    require(
        len(args.expected_token_span_fingerprint) == 64,
        "--expected-token-span-fingerprint must be a SHA256",
    )
    require(0.0 < args.gpu_memory_utilization <= 1.0, "invalid GPU memory utilization")
    require(args.cpu_offload_gb >= 0.0, "CPU offload must be non-negative")
    require(args.max_model_len >= args.expected_prompt_tokens + 1, "max model length is too small")


def main() -> int:
    args = parser().parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        validate_args(args)
        manifest = collect(args)
        print(f"{PASS_LINE}: {manifest.parent}")
        return 0
    except (ReferenceError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"FERRUM C13 VLLM RAW LOGITS REFERENCE REJECT: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
