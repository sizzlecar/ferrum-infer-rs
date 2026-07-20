#!/usr/bin/env python3
"""Build an independent FP32 Qwen3.5 layer-0 linear-attention reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePath
from typing import Any


PASS_PREFIX = "QWEN35 GGUF LINEAR ATTENTION REFERENCE PASS"
SELF_TEST_PASS = "QWEN35 GGUF LINEAR ATTENTION REFERENCE SELF-TEST PASS"
VALUE_ID = "value.layer.0.attention"
HIDDEN_SIZE = 2560
VOCABULARY_SIZE = 248320
KEY_HEADS = 16
VALUE_HEADS = 32
HEAD_DIM = 128
QK_SIZE = KEY_HEADS * HEAD_DIM
VALUE_SIZE = VALUE_HEADS * HEAD_DIM
QKV_SIZE = QK_SIZE * 2 + VALUE_SIZE
CONV_KERNEL = 4
RMS_EPSILON = 1.0e-6
RELATIVE_ERROR_FLOOR = 1.0e-12
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_METADATA: dict[str, Any] = {
    "general.architecture": "qwen35",
    "general.basename": "Qwen3.5-4B",
    "general.file_type": 15,
    "general.quantization_version": 2,
    "qwen35.attention.head_count": 16,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.layer_norm_rms_epsilon": RMS_EPSILON,
    "qwen35.attention.value_length": 256,
    "qwen35.block_count": 32,
    "qwen35.embedding_length": HIDDEN_SIZE,
    "qwen35.full_attention_interval": 4,
    "qwen35.ssm.conv_kernel": CONV_KERNEL,
    "qwen35.ssm.group_count": KEY_HEADS,
    "qwen35.ssm.inner_size": VALUE_SIZE,
    "qwen35.ssm.state_size": HEAD_DIM,
    "qwen35.ssm.time_step_rank": VALUE_HEADS,
}

EXPECTED_TENSORS: dict[str, tuple[tuple[int, ...], str]] = {
    "token_embd.weight": ((HIDDEN_SIZE, VOCABULARY_SIZE), "Q6_K"),
    "blk.0.attn_norm.weight": ((HIDDEN_SIZE,), "F32"),
    "blk.0.attn_qkv.weight": ((HIDDEN_SIZE, QKV_SIZE), "Q5_K"),
    "blk.0.attn_gate.weight": ((HIDDEN_SIZE, VALUE_SIZE), "Q4_K"),
    "blk.0.ssm_beta.weight": ((HIDDEN_SIZE, VALUE_HEADS), "Q8_0"),
    "blk.0.ssm_alpha.weight": ((HIDDEN_SIZE, VALUE_HEADS), "Q8_0"),
    "blk.0.ssm_conv1d.weight": ((CONV_KERNEL, QKV_SIZE), "F32"),
    "blk.0.ssm_a": ((VALUE_HEADS,), "F32"),
    "blk.0.ssm_dt.bias": ((VALUE_HEADS,), "F32"),
    "blk.0.ssm_norm.weight": ((HEAD_DIM,), "F32"),
    "blk.0.ssm_out.weight": ((VALUE_SIZE, HIDDEN_SIZE), "Q5_K"),
}


class ReferenceError(RuntimeError):
    """The input or generated reference artifact is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReferenceError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ReferenceError(f"non-finite JSON constant is forbidden: {value}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except ReferenceError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferenceError(f"cannot load {path}: {error}") from error


def write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ReferenceError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256_bytes(payload)


def safe_child(root: Path, raw_name: Any, label: str) -> Path:
    require(isinstance(raw_name, str) and raw_name == raw_name.strip(),
            f"{label} must be a trimmed string")
    pure = PurePath(raw_name)
    require(bool(raw_name) and not pure.is_absolute(), f"{label} must be relative")
    require(".." not in pure.parts, f"{label} cannot escape its artifact directory")
    path = root / pure
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    return path


def command_output(command: list[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise ReferenceError(f"cannot run {' '.join(command)}: {error}") from error
    require(completed.returncode == 0,
            f"{' '.join(command)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def git_provenance(root: Path, *, allow_untracked: bool) -> dict[str, Any]:
    root = root.resolve()
    sha = command_output(["git", "rev-parse", "HEAD"], root)
    require(re.fullmatch(r"[0-9a-f]{40}", sha) is not None,
            f"invalid Git SHA for {root}")
    command = ["git", "status", "--short"]
    if allow_untracked:
        command.append("--untracked-files=no")
    status = command_output(command, root)
    require(not status, f"tracked oracle source is dirty under {root}: {status}")
    return {"root": str(root), "git_sha": sha, "tracked_dirty": False}


def read_optional_text(root: Path, name: str) -> str | None:
    path = root / name
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise ReferenceError(f"cannot read {path}: {error}") from error


def load_token_ids(path: Path) -> list[int]:
    document = load_json(path)
    require(isinstance(document, dict), "prompt token JSON must be an object")
    allowed = {
        "schema_version",
        "request_id",
        "model",
        "tokenizer_or_model",
        "token_count",
        "token_ids",
        "unavailable_reason",
        "sanitized",
    }
    require(not set(document) - allowed,
            f"prompt token JSON has unknown fields: {sorted(set(document) - allowed)}")
    if "schema_version" in document:
        require(document["schema_version"] == 1,
                "prompt token JSON schema_version must be 1")
        require(document.get("model") == "qwen3.5:4b-q4_k_m",
                "prompt token JSON model is not the reviewed fixture")
        require(document.get("tokenizer_or_model") == document["model"],
                "prompt token JSON tokenizer/model identity differs")
        require(document.get("unavailable_reason") is None,
                "prompt token IDs were marked unavailable")
        require(document.get("sanitized") is True,
                "prompt token JSON must be sanitized")
    values = document.get("token_ids")
    require(isinstance(values, list) and values, "token_ids must be a non-empty list")
    token_ids: list[int] = []
    for index, value in enumerate(values):
        require(isinstance(value, int) and not isinstance(value, bool),
                f"token_ids[{index}] must be an integer")
        require(0 <= value < VOCABULARY_SIZE,
                f"token_ids[{index}] is outside the Qwen3.5 vocabulary")
        token_ids.append(value)
    require(len(token_ids) <= 128, "reference fixture cannot exceed 128 tokens")
    if "token_count" in document:
        require(document["token_count"] == len(token_ids),
                "token_count does not match token_ids")
    return token_ids


def load_ferrum_checkpoint(
    capture_dir: Path, token_count: int
) -> tuple[Path, dict[str, Any]]:
    plan_path = capture_dir / "plan.json"
    wave_paths = sorted(capture_dir.glob("wave-*.json"))
    require(plan_path.is_file(), f"missing Ferrum capture plan: {plan_path}")
    require(len(wave_paths) == 1, "expected exactly one Ferrum capture wave")
    plan = load_json(plan_path)
    wave = load_json(wave_paths[0])
    require(isinstance(plan, dict) and isinstance(wave, dict),
            "Ferrum plan and wave must be JSON objects")
    require(plan.get("schema_version") == 1, "unsupported Ferrum plan schema")
    require(wave.get("schema_version") == 1, "unsupported Ferrum wave schema")
    require(wave.get("wave_kind") == "prefill", "Ferrum checkpoint must be prefill")
    require(wave.get("participant_count") == 1,
            "Ferrum checkpoint must contain exactly one participant")
    for key in ("plan_id", "plan_hash", "model_id", "family_fingerprint",
                "program_fingerprint", "run_id"):
        require(plan.get(key) == wave.get(key), f"Ferrum {key} differs across plan/wave")
    require(plan.get("model_id") == "Qwen3.5-4B-Q4_K_M",
            "Ferrum capture model_id is not the reviewed Qwen3.5 fixture")

    records = wave.get("records")
    require(isinstance(records, list), "Ferrum wave records must be a list")
    selected = [
        record
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("value"), dict)
        and record["value"].get("value_id") == VALUE_ID
    ]
    require(len(selected) == 1, f"Ferrum wave must contain exactly one {VALUE_ID}")
    record = selected[0]
    require(record.get("participant_index") == 0,
            "Ferrum checkpoint participant index must be zero")
    span = record.get("token_span")
    require(isinstance(span, dict), "Ferrum checkpoint token_span must be an object")
    require(span.get("immediate_tokens") == token_count,
            "Ferrum checkpoint token count differs from the oracle fixture")
    require(span.get("full_input_tokens") == token_count,
            "Ferrum checkpoint must cover the full input")
    require(span.get("immediate_start_token") == 0
            and span.get("immediate_end_token") == token_count,
            "Ferrum checkpoint token range is not the full prompt")

    value = record["value"]
    tensor = value.get("tensor")
    layout = record.get("output_layout")
    require(isinstance(tensor, dict) and isinstance(layout, dict),
            "Ferrum checkpoint tensor/layout must be objects")
    require(tensor.get("element_type") == "f16"
            and layout.get("element_type") == "f16",
            "Ferrum layer checkpoint must be logical fp16")
    dimensions = tensor.get("dimensions")
    require(isinstance(dimensions, list) and len(dimensions) == 2,
            "Ferrum checkpoint must have a rank-2 capacity shape")
    require(dimensions[0] >= token_count and dimensions[1] == HIDDEN_SIZE,
            "Ferrum checkpoint capacity shape is incompatible with the fixture")
    element_count = token_count * HIDDEN_SIZE
    require(layout.get("element_count") == element_count,
            "Ferrum checkpoint element count differs from [tokens, hidden]")
    require(record.get("raw_bytes") == element_count * 2,
            "Ferrum checkpoint raw byte count is invalid")
    raw_path = safe_child(capture_dir, record.get("raw_file"), "Ferrum raw_file")
    require(raw_path.stat().st_size == element_count * 2,
            "Ferrum checkpoint file size is invalid")
    raw_sha = sha256_file(raw_path)
    require(SHA256_RE.fullmatch(str(record.get("raw_sha256", ""))) is not None,
            "Ferrum checkpoint raw_sha256 is invalid")
    require(raw_sha == record["raw_sha256"], "Ferrum checkpoint SHA256 mismatch")

    artifact_root = capture_dir.parent
    provenance = {
        "artifact_root": str(artifact_root.resolve()),
        "plan_file": plan_path.name,
        "plan_sha256": sha256_file(plan_path),
        "wave_file": wave_paths[0].name,
        "wave_sha256": sha256_file(wave_paths[0]),
        "git_sha": read_optional_text(artifact_root, "git-sha.txt"),
        "git_status": read_optional_text(artifact_root, "git-status.txt"),
        "tracked_diff_sha256": (
            sha256_file(artifact_root / "tracked-diff.patch")
            if (artifact_root / "tracked-diff.patch").is_file()
            else None
        ),
        "binary_sha256_record": read_optional_text(artifact_root, "binary-sha256.txt"),
        "value_id": VALUE_ID,
        "logical_dtype": "fp16",
        "logical_shape": [token_count, HIDDEN_SIZE],
        "raw_file": raw_path.name,
        "raw_sha256": raw_sha,
    }
    return raw_path, provenance


def load_llama_checkpoint(
    checkpoint_dir: Path, token_count: int
) -> tuple[Path, dict[str, Any]]:
    manifest_path = checkpoint_dir / "checkpoint-manifest.json"
    manifest = load_json(manifest_path)
    require(isinstance(manifest, dict) and manifest.get("schema_version") == 1,
            "unsupported llama checkpoint manifest")
    require(manifest.get("status") == "pass" and manifest.get("output_dtype") == "f32",
            "llama checkpoint manifest is not a successful FP32 capture")
    records = manifest.get("records")
    require(isinstance(records, list), "llama checkpoint records must be a list")
    selected = [
        record for record in records
        if isinstance(record, dict) and record.get("tensor_name") == "attn_residual-0"
    ]
    require(len(selected) == 1, "llama checkpoint must contain one attn_residual-0")
    record = selected[0]
    expected_shape = [token_count, HIDDEN_SIZE]
    require(record.get("logical_shape") == expected_shape,
            "llama attention residual shape differs from the fixture")
    require(record.get("element_count") == token_count * HIDDEN_SIZE,
            "llama attention residual element count is invalid")
    raw_path = safe_child(checkpoint_dir, record.get("raw_file"), "llama raw_file")
    require(raw_path.stat().st_size == token_count * HIDDEN_SIZE * 4,
            "llama attention residual byte count is invalid")
    return raw_path, {
        "manifest_sha256": sha256_file(manifest_path),
        "logical_dtype": "fp32",
        "logical_shape": expected_shape,
        "raw_file": raw_path.name,
        "raw_sha256": sha256_file(raw_path),
    }


def json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if hasattr(value, "item"):
        return json_value(value.item())
    raise ReferenceError(f"GGUF metadata value is not JSON-compatible: {type(value)}")


def hf_snapshot_identity(path: Path) -> dict[str, str | None]:
    parts = path.resolve().parts
    result: dict[str, str | None] = {"repository": None, "revision": None}
    for index, part in enumerate(parts):
        if part.startswith("models--"):
            result["repository"] = part.removeprefix("models--").replace("--", "/")
        if part == "snapshots" and index + 1 < len(parts):
            revision = parts[index + 1]
            if re.fullmatch(r"[0-9a-f]{40}", revision):
                result["revision"] = revision
    return result


def measure(np: Any, actual: Any, expected: Any) -> dict[str, Any]:
    actual64 = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected64 = np.asarray(expected, dtype=np.float64).reshape(-1)
    require(actual64.shape == expected64.shape, "metric tensor shapes differ")
    actual_nan = int(np.isnan(actual64).sum())
    actual_inf = int(np.isinf(actual64).sum())
    expected_nan = int(np.isnan(expected64).sum())
    expected_inf = int(np.isinf(expected64).sum())
    require(actual_nan == 0 and actual_inf == 0, "actual tensor contains NaN or Inf")
    require(expected_nan == 0 and expected_inf == 0,
            "reference tensor contains NaN or Inf")
    difference = actual64 - expected64
    absolute = np.abs(difference)
    expected_norm = float(np.linalg.norm(expected64))
    actual_norm = float(np.linalg.norm(actual64))
    require(expected_norm > 0.0 and actual_norm > 0.0,
            "metric tensor norm must be positive")
    cosine = float(np.dot(actual64, expected64) / (actual_norm * expected_norm))
    return {
        "element_count": int(actual64.size),
        "actual_nan_count": actual_nan,
        "actual_inf_count": actual_inf,
        "expected_nan_count": expected_nan,
        "expected_inf_count": expected_inf,
        "max_abs": float(np.max(absolute)),
        "max_relative_error": float(
            np.max(absolute / np.maximum(np.abs(expected64), RELATIVE_ERROR_FLOOR))
        ),
        "max_relative_error_denominator_floor": RELATIVE_ERROR_FLOOR,
        "relative_l2": float(np.linalg.norm(difference) / expected_norm),
        "cosine": max(-1.0, min(1.0, cosine)),
    }


def stable_sigmoid(np: Any, value: Any) -> Any:
    output = np.empty_like(value, dtype=np.float32)
    positive = value >= 0
    output[positive] = np.reciprocal(
        np.float32(1.0) + np.exp(-value[positive], dtype=np.float32)
    )
    negative_exp = np.exp(value[~positive], dtype=np.float32)
    output[~positive] = negative_exp / (np.float32(1.0) + negative_exp)
    return output


def stable_softplus(np: Any, value: Any) -> Any:
    output = np.empty_like(value, dtype=np.float32)
    high = value > np.float32(20.0)
    low = value < np.float32(-20.0)
    middle = ~(high | low)
    output[high] = value[high]
    output[low] = np.exp(value[low], dtype=np.float32)
    output[middle] = np.log1p(np.exp(value[middle], dtype=np.float32),
                              dtype=np.float32)
    return output


def build_reference(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    source = git_provenance(repo_root, allow_untracked=True)
    llama_root = Path(args.llama_cpp_root).expanduser().resolve()
    require((llama_root / "gguf-py/gguf").is_dir(),
            "--llama-cpp-root does not contain gguf-py/gguf")
    llama_source = git_provenance(llama_root, allow_untracked=False)
    sys.path.insert(0, str(llama_root / "gguf-py"))
    try:
        import numpy as np  # type: ignore
        import gguf  # type: ignore
        from gguf import GGUFReader  # type: ignore
        from gguf.quants import dequantize  # type: ignore
    except ImportError as error:
        raise ReferenceError(
            "numpy and llama.cpp gguf-py are required; run with "
            "`uv run --with numpy --with pyyaml`"
        ) from error

    model_input_path = Path(args.model).expanduser().absolute()
    model_path = model_input_path.resolve()
    token_path = Path(args.prompt_token_ids).expanduser().resolve()
    capture_dir = Path(args.ferrum_capture).expanduser().resolve()
    require(model_path.is_file() and not model_path.is_symlink(),
            "--model must be a regular GGUF file")
    require(token_path.is_file() and not token_path.is_symlink(),
            "--prompt-token-ids must be a regular JSON file")
    require(capture_dir.is_dir(), "--ferrum-capture must be a directory")
    token_ids = load_token_ids(token_path)
    token_count = len(token_ids)
    actual_path, actual_provenance = load_ferrum_checkpoint(capture_dir, token_count)

    reader = GGUFReader(str(model_path), "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    require(len(tensors) == len(reader.tensors), "GGUF contains duplicate tensor names")

    metadata: dict[str, Any] = {}
    for key, expected in EXPECTED_METADATA.items():
        field = reader.get_field(key)
        require(field is not None, f"GGUF is missing metadata: {key}")
        actual = json_value(field.contents())
        if isinstance(expected, float):
            require(math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12),
                    f"GGUF metadata {key} differs: {actual} != {expected}")
        else:
            require(actual == expected,
                    f"GGUF metadata {key} differs: {actual} != {expected}")
        metadata[key] = actual
    for key in sorted(reader.fields):
        if key.startswith(("general.", "qwen35.", "quantize.")):
            metadata[key] = json_value(reader.fields[key].contents())
    write_json(out_dir / "gguf-metadata.json", metadata)

    inventory = []
    for tensor in sorted(reader.tensors, key=lambda item: item.name):
        inventory.append(
            {
                "name": tensor.name,
                "shape": [int(value) for value in tensor.shape],
                "tensor_type": tensor.tensor_type.name,
                "element_count": int(tensor.n_elements),
                "byte_count": int(tensor.n_bytes),
            }
        )
    write_json(out_dir / "tensor-inventory.json", inventory)
    for name, (expected_shape, expected_type) in EXPECTED_TENSORS.items():
        require(name in tensors, f"GGUF is missing tensor: {name}")
        tensor = tensors[name]
        shape = tuple(int(value) for value in tensor.shape)
        require(shape == expected_shape,
                f"GGUF tensor {name} shape differs: {shape} != {expected_shape}")
        require(tensor.tensor_type.name == expected_type,
                f"GGUF tensor {name} type differs: "
                f"{tensor.tensor_type.name} != {expected_type}")

    def weight(name: str) -> Any:
        tensor = tensors[name]
        value = dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False)
        require(bool(np.isfinite(value).all()), f"dequantized tensor is non-finite: {name}")
        return value

    token_index = np.asarray(token_ids, dtype=np.int64)
    embedding_tensor = tensors["token_embd.weight"]
    embedding = dequantize(
        embedding_tensor.data[token_index].copy(), embedding_tensor.tensor_type
    ).astype(np.float32, copy=False)
    require(embedding.shape == (token_count, HIDDEN_SIZE),
            "dequantized embedding shape differs from the fixture")

    norm_weight = weight("blk.0.attn_norm.weight")
    inverse = np.reciprocal(
        np.sqrt(
            np.mean(embedding * embedding, axis=1, dtype=np.float32)
            + np.float32(RMS_EPSILON),
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    normalized = embedding * inverse[:, None] * norm_weight[None, :]

    qkv_weight = weight("blk.0.attn_qkv.weight")
    qkv = normalized @ qkv_weight.T
    del qkv_weight
    gate_weight = weight("blk.0.attn_gate.weight")
    z = normalized @ gate_weight.T
    del gate_weight
    beta_weight = weight("blk.0.ssm_beta.weight")
    beta_raw = normalized @ beta_weight.T
    del beta_weight
    alpha_weight = weight("blk.0.ssm_alpha.weight")
    alpha_raw = normalized @ alpha_weight.T
    del alpha_weight

    conv_weight = weight("blk.0.ssm_conv1d.weight")
    padded = np.zeros((token_count + CONV_KERNEL - 1, QKV_SIZE), dtype=np.float32)
    padded[CONV_KERNEL - 1:] = qkv
    conv_raw = np.empty((token_count, QKV_SIZE), dtype=np.float32)
    for token in range(token_count):
        conv_raw[token] = np.sum(
            padded[token:token + CONV_KERNEL] * conv_weight.T,
            axis=0,
            dtype=np.float32,
        )
    conv_output = conv_raw * stable_sigmoid(np, conv_raw)

    query = conv_output[:, :QK_SIZE].reshape(token_count, KEY_HEADS, HEAD_DIM).copy()
    key = conv_output[:, QK_SIZE:2 * QK_SIZE].reshape(
        token_count, KEY_HEADS, HEAD_DIM
    ).copy()
    value = conv_output[:, 2 * QK_SIZE:].reshape(
        token_count, VALUE_HEADS, HEAD_DIM
    ).copy()
    query *= np.reciprocal(
        np.sqrt(
            np.sum(query * query, axis=2, dtype=np.float32)
            + np.float32(RMS_EPSILON),
            dtype=np.float32,
        ),
        dtype=np.float32,
    )[:, :, None]
    key *= np.reciprocal(
        np.sqrt(
            np.sum(key * key, axis=2, dtype=np.float32) + np.float32(RMS_EPSILON),
            dtype=np.float32,
        ),
        dtype=np.float32,
    )[:, :, None]

    log_decay = weight("blk.0.ssm_a")
    time_bias = weight("blk.0.ssm_dt.bias")
    decay_gate = log_decay[None, :] * stable_softplus(np, alpha_raw + time_bias[None, :])
    beta = stable_sigmoid(np, beta_raw)

    state = np.zeros((VALUE_HEADS, HEAD_DIM, HEAD_DIM), dtype=np.float32)
    recurrent = np.empty((token_count, VALUE_HEADS, HEAD_DIM), dtype=np.float32)
    head_map = np.arange(VALUE_HEADS) % KEY_HEADS
    scale = np.float32(1.0 / math.sqrt(HEAD_DIM))
    for token in range(token_count):
        token_query = query[token, head_map]
        token_key = key[token, head_map]
        state *= np.exp(decay_gate[token], dtype=np.float32)[:, None, None]
        predicted = np.einsum(
            "hvk,hk->hv", state, token_key, dtype=np.float32, optimize=False
        )
        delta = (value[token] - predicted) * beta[token, :, None]
        state += delta[:, :, None] * token_key[:, None, :]
        recurrent[token] = np.einsum(
            "hvk,hk->hv", state, token_query, dtype=np.float32, optimize=False
        ) * scale

    recurrent_norm_weight = weight("blk.0.ssm_norm.weight")
    recurrent_inverse = np.reciprocal(
        np.sqrt(
            np.mean(recurrent * recurrent, axis=2, dtype=np.float32)
            + np.float32(RMS_EPSILON),
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    gated = (
        recurrent
        * recurrent_inverse[:, :, None]
        * recurrent_norm_weight[None, None, :]
        * (z * stable_sigmoid(np, z)).reshape(token_count, VALUE_HEADS, HEAD_DIM)
    ).reshape(token_count, VALUE_SIZE)
    output_weight = weight("blk.0.ssm_out.weight")
    reference = embedding + gated @ output_weight.T
    require(reference.shape == (token_count, HIDDEN_SIZE),
            "reference residual shape differs from the fixture")
    require(reference.dtype == np.float32, "reference residual is not FP32")
    require(bool(np.isfinite(reference).all()), "reference residual contains NaN or Inf")

    actual = np.fromfile(actual_path, dtype="<f2").reshape(token_count, HIDDEN_SIZE)
    reference_path = out_dir / "layer-0-linear-attention-residual.f32"
    np.asarray(reference, dtype="<f4").tofile(reference_path)
    actual_f32 = np.asarray(actual, dtype="<f4")
    reference_f32 = np.asarray(reference, dtype="<f4")
    metrics = measure(np, actual_f32, reference_f32)
    metrics.update(
        {
            "shape": [token_count, HIDDEN_SIZE],
            "actual_logical_dtype": "fp16",
            "oracle_precision": "fp32",
            "actual_f32_sha256": sha256_bytes(actual_f32.tobytes(order="C")),
            "expected_f32_sha256": sha256_file(reference_path),
        }
    )

    cross_validation = None
    if args.llama_checkpoints:
        llama_dir = Path(args.llama_checkpoints).expanduser().resolve()
        llama_path, llama_provenance = load_llama_checkpoint(llama_dir, token_count)
        llama_residual = np.fromfile(llama_path, dtype="<f4").reshape(
            token_count, HIDDEN_SIZE
        )
        cross_validation = {
            "role": "same-GGUF quantized implementation cross-validation only",
            "provenance": llama_provenance,
            "metrics": measure(np, llama_residual, reference_f32),
        }

    model_sha = sha256_file(model_path)
    tensor_inventory_path = out_dir / "tensor-inventory.json"
    metadata_path = out_dir / "gguf-metadata.json"
    prompt_sequence_sha = sha256_bytes(
        b"".join(int(value).to_bytes(4, "little") for value in token_ids)
    )
    report = {
        "schema_version": 1,
        "status": "measured",
        "oracle": {
            "identity": "cpu.fp32.python.qwen35_gguf_linear_attention_reference",
            "precision": "fp32",
            "semantics": (
                "independent Qwen3.5 layer-0 linear-attention execution over "
                "gguf-py-dequantized weights"
            ),
            "source_path": str(Path(__file__).resolve().relative_to(repo_root)),
            "source_sha256": sha256_file(Path(__file__).resolve()),
            "ferrum_source": source,
            "llama_cpp_gguf_py_source": llama_source,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "gguf_module_path": str(Path(gguf.__file__).resolve()),
        },
        "model": {
            "path": str(model_path),
            "sha256": model_sha,
            "byte_count": model_path.stat().st_size,
            "format": "GGUF Q4_K_M",
            "hugging_face_snapshot": hf_snapshot_identity(model_input_path),
            "upstream_model": metadata.get("general.base_model.0.repo_url"),
            "upstream_revision": "not_declared_in_gguf",
            "converter_version": "not_declared_in_gguf",
            "quantized_by": metadata.get("general.quantized_by"),
            "quantization_version": metadata["general.quantization_version"],
            "license": metadata.get("general.license"),
            "metadata_file": metadata_path.name,
            "metadata_sha256": sha256_file(metadata_path),
            "tensor_inventory_file": tensor_inventory_path.name,
            "tensor_inventory_sha256": sha256_file(tensor_inventory_path),
            "tensor_count": len(inventory),
        },
        "fixture": {
            "fixture_id": (
                "qwen35-4b.gguf-q4-k-m.layer-0.linear-attention."
                f"tokens-{token_count}"
            ),
            "layer_index": 0,
            "token_ids": token_ids,
            "token_sequence_sha256": prompt_sequence_sha,
            "shape": [token_count, HIDDEN_SIZE],
            "value_head_mapping": "interleaved_by_key_head",
            "decay_parameterization": "negative_rate",
        },
        "actual": actual_provenance,
        "reference": {
            "raw_file": reference_path.name,
            "raw_sha256": sha256_file(reference_path),
            "logical_dtype": "fp32",
            "logical_shape": [token_count, HIDDEN_SIZE],
        },
        "metrics": metrics,
        "llama_cpp_cross_validation": cross_validation,
        "invocation": {
            "argv": [str(value) for value in sys.argv],
            "cwd": os.getcwd(),
        },
    }
    write_json(out_dir / "report.json", report)
    return report


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="qwen35-linear-reference-") as tmp:
        root = Path(tmp)
        valid = root / "tokens.json"
        write_json(
            valid,
            {
                "schema_version": 1,
                "request_id": "request-1",
                "model": "qwen3.5:4b-q4_k_m",
                "tokenizer_or_model": "qwen3.5:4b-q4_k_m",
                "token_count": 2,
                "token_ids": [0, VOCABULARY_SIZE - 1],
                "unavailable_reason": None,
                "sanitized": True,
            },
        )
        require(load_token_ids(valid) == [0, VOCABULARY_SIZE - 1],
                "valid token fixture did not roundtrip")

        duplicate = root / "duplicate.json"
        duplicate.write_text('{"token_ids":[1],"token_ids":[2]}\n', encoding="utf-8")
        try:
            load_json(duplicate)
        except ReferenceError as error:
            require("duplicate JSON key" in str(error), "wrong duplicate-key failure")
        else:
            raise ReferenceError("duplicate JSON keys unexpectedly passed")

        raw = root / "actual.bin"
        raw.write_bytes(b"ok")
        require(safe_child(root, "actual.bin", "raw") == raw,
                "safe artifact child did not resolve")
        try:
            safe_child(root, "../actual.bin", "raw")
        except ReferenceError as error:
            require("cannot escape" in str(error), "wrong traversal failure")
        else:
            raise ReferenceError("artifact path traversal unexpectedly passed")

        identity = hf_snapshot_identity(
            Path("/cache/models--owner--model/snapshots/" + "a" * 40 + "/model.gguf")
        )
        require(identity == {"repository": "owner/model", "revision": "a" * 40},
                "Hugging Face snapshot identity parsing failed")
    print(SELF_TEST_PASS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--prompt-token-ids")
    parser.add_argument("--ferrum-capture")
    parser.add_argument("--llama-cpp-root")
    parser.add_argument("--llama-checkpoints")
    parser.add_argument("--out")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    required = {
        "--model": args.model,
        "--prompt-token-ids": args.prompt_token_ids,
        "--ferrum-capture": args.ferrum_capture,
        "--llama-cpp-root": args.llama_cpp_root,
        "--out": args.out,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        print(f"missing required arguments: {', '.join(missing)}", file=sys.stderr)
        return 2

    out_dir = Path(args.out).expanduser().resolve()
    try:
        require(not out_dir.exists() or out_dir.is_dir(),
                f"output path is not a directory: {out_dir}")
        require(not out_dir.exists() or not any(out_dir.iterdir()),
                f"output directory is not empty: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        build_reference(args, out_dir)
    except Exception as error:
        out_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": 1,
            "status": "fail",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        try:
            write_json(out_dir / "failure.json", failure)
        except OSError:
            pass
        print(f"QWEN35 GGUF LINEAR ATTENTION REFERENCE FAIL: {error}", file=sys.stderr)
        return 1
    print(f"{PASS_PREFIX}: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
