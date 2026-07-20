#!/usr/bin/env python3
"""Build an independent FP32 Qwen3.5 full-model and full-vocabulary reference."""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import platform
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import qwen35_gguf_full_attention_reference as full_attention
import qwen35_gguf_linear_attention_reference as common


PASS_PREFIX = "QWEN35 GGUF MODEL REFERENCE PASS"
SELF_TEST_PASS = "QWEN35 GGUF MODEL REFERENCE SELF-TEST PASS"
HIDDEN_SIZE = 2560
VOCABULARY_SIZE = 248320
INTERMEDIATE_SIZE = 9216
LAYER_COUNT = 32
FULL_ATTENTION_INTERVAL = 4
KEY_HEADS = 16
VALUE_HEADS = 32
HEAD_DIM = 128
QK_SIZE = KEY_HEADS * HEAD_DIM
VALUE_SIZE = VALUE_HEADS * HEAD_DIM
QKV_SIZE = QK_SIZE * 2 + VALUE_SIZE
CONV_KERNEL = 4
RMS_EPSILON = 1.0e-6
LOGIT_CHUNK_ROWS = 4096
EMBEDDING_VALUE_ID = "value.hidden.embedding"
FINAL_HIDDEN_VALUE_ID = "value.output.final_hidden"
LOGITS_VALUE_ID = "value.output.logits"
EXTRACTOR_SOURCE_COMMIT = "7a75cdcd6db7fb427d3b175200ef218eff364e05"
EXTRACTOR_SOURCE_SHA256 = "2ae2f57bfcd38d69d046e8d4dadb545f8423e72d9af46cd2836490aee2662253"


def layer_value_id(layer: int) -> str:
    return f"value.layer.{layer}.output"


def load_ferrum_capture(
    capture_dir: Path, token_count: int
) -> tuple[dict[str, Path], dict[str, Any]]:
    plan_path = capture_dir / "plan.json"
    wave_paths = sorted(capture_dir.glob("wave-*.json"))
    common.require(plan_path.is_file(), f"missing Ferrum capture plan: {plan_path}")
    common.require(len(wave_paths) == 1, "expected one Ferrum capture wave")
    plan = common.load_json(plan_path)
    wave = common.load_json(wave_paths[0])
    common.require(isinstance(plan, dict) and isinstance(wave, dict),
                   "Ferrum capture plan/wave must be objects")
    common.require(plan.get("schema_version") == 1 and wave.get("schema_version") == 1,
                   "unsupported Ferrum capture schema")
    common.require(wave.get("wave_kind") == "prefill"
                   and wave.get("participant_count") == 1,
                   "Ferrum capture must be one-participant prefill")
    for key in (
        "plan_id",
        "plan_hash",
        "model_id",
        "family_fingerprint",
        "program_fingerprint",
        "run_id",
    ):
        common.require(plan.get(key) == wave.get(key),
                       f"Ferrum {key} differs across plan/wave")
    common.require(plan.get("model_id") == "Qwen3.5-4B-Q4_K_M",
                   "Ferrum capture model differs from the reviewed fixture")
    records = wave.get("records")
    common.require(isinstance(records, list), "Ferrum capture records must be a list")

    expected = [EMBEDDING_VALUE_ID]
    expected.extend(layer_value_id(layer) for layer in range(LAYER_COUNT))
    expected.extend((FINAL_HIDDEN_VALUE_ID, LOGITS_VALUE_ID))
    paths: dict[str, Path] = {}
    checkpoints: dict[str, Any] = {}
    for value_id in expected:
        selected = [
            record
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("value"), dict)
            and record["value"].get("value_id") == value_id
        ]
        common.require(len(selected) == 1,
                       f"Ferrum wave must contain exactly one {value_id}")
        record = selected[0]
        common.require(record.get("participant_index") == 0,
                       f"{value_id} participant index must be zero")
        span = record.get("token_span")
        common.require(
            isinstance(span, dict)
            and span.get("immediate_tokens") == token_count
            and span.get("full_input_tokens") == token_count
            and span.get("immediate_start_token") == 0
            and span.get("immediate_end_token") == token_count,
            f"{value_id} does not cover the complete prompt",
        )
        tensor = record["value"].get("tensor")
        layout = record.get("output_layout")
        common.require(isinstance(tensor, dict) and isinstance(layout, dict),
                       f"{value_id} tensor/layout must be objects")
        common.require(tensor.get("element_type") == "f16"
                       and layout.get("element_type") == "f16",
                       f"{value_id} must be logical FP16")
        if value_id == LOGITS_VALUE_ID:
            shape = [VOCABULARY_SIZE]
            dimensions = [1, VOCABULARY_SIZE]
            common.require(tensor.get("dimensions") == dimensions,
                           f"{value_id} dimensions differ from the fixture")
        else:
            shape = [token_count, HIDDEN_SIZE]
            dimensions = tensor.get("dimensions")
            common.require(
                isinstance(dimensions, list)
                and len(dimensions) == 2
                and dimensions[0] >= token_count
                and dimensions[1] == HIDDEN_SIZE,
                f"{value_id} capacity differs from the fixture",
            )
        elements = math.prod(shape)
        common.require(layout.get("element_count") == elements
                       and record.get("raw_bytes") == elements * 2,
                       f"{value_id} element/byte count differs")
        raw_path = common.safe_child(capture_dir, record.get("raw_file"),
                                     f"{value_id}.raw_file")
        common.require(raw_path.stat().st_size == elements * 2,
                       f"{value_id} file size differs")
        raw_sha = common.sha256_file(raw_path)
        common.require(raw_sha == record.get("raw_sha256"),
                       f"{value_id} SHA256 mismatch")
        paths[value_id] = raw_path
        checkpoints[value_id] = {
            "logical_dtype": "fp16",
            "logical_shape": shape,
            "raw_file": raw_path.name,
            "raw_sha256": raw_sha,
        }

    root = capture_dir.parent
    provenance = {
        "artifact_root": str(root.resolve()),
        "plan_file": plan_path.name,
        "plan_sha256": common.sha256_file(plan_path),
        "wave_file": wave_paths[0].name,
        "wave_sha256": common.sha256_file(wave_paths[0]),
        "git_sha": common.read_optional_text(root, "git-sha.txt"),
        "git_status": common.read_optional_text(root, "git-status.txt"),
        "tracked_diff_sha256": (
            common.sha256_file(root / "tracked-diff.patch")
            if (root / "tracked-diff.patch").is_file()
            else None
        ),
        "binary_sha256_record": common.read_optional_text(root, "binary-sha256.txt"),
        "checkpoints": checkpoints,
    }
    return paths, provenance


def load_llama_artifact(
    artifact_root: Path,
    extractor_root: Path,
    token_count: int,
) -> tuple[dict[str, Path], dict[str, Any]]:
    checkpoint_dir = artifact_root / "checkpoints"
    manifest_path = checkpoint_dir / "checkpoint-manifest.json"
    manifest = common.load_json(manifest_path)
    common.require(isinstance(manifest, dict) and manifest.get("schema_version") == 1,
                   "unsupported llama checkpoint manifest")
    common.require(manifest.get("status") == "pass"
                   and manifest.get("output_dtype") == "f32",
                   "llama checkpoint manifest is not successful FP32")
    records = manifest.get("records")
    common.require(isinstance(records, list), "llama checkpoint records must be a list")
    expected = ["model.input_embed"]
    expected.extend(f"l_out-{layer}" for layer in range(LAYER_COUNT))
    expected.extend(("result_norm", "result_output"))
    common.require(
        sorted(record.get("tensor_name") for record in records if isinstance(record, dict))
        == sorted(expected),
        "llama checkpoint names differ from the reviewed matrix",
    )
    paths: dict[str, Path] = {}
    checkpoint_provenance: dict[str, Any] = {}
    for name in expected:
        selected = [
            record for record in records
            if isinstance(record, dict) and record.get("tensor_name") == name
        ]
        common.require(len(selected) == 1, f"llama artifact must contain one {name}")
        record = selected[0]
        if name == "result_output":
            shape = [VOCABULARY_SIZE]
        elif name in ("l_out-31", "result_norm"):
            shape = [HIDDEN_SIZE]
        else:
            shape = [token_count, HIDDEN_SIZE]
        common.require(record.get("logical_shape") == shape
                       and record.get("element_count") == math.prod(shape),
                       f"llama {name} shape differs")
        raw = common.safe_child(checkpoint_dir, record.get("raw_file"),
                                f"llama.{name}.raw_file")
        common.require(raw.stat().st_size == math.prod(shape) * 4,
                       f"llama {name} byte count differs")
        paths[name] = raw
        checkpoint_provenance[name] = {
            "logical_dtype": "fp32",
            "logical_shape": shape,
            "raw_file": raw.name,
            "raw_sha256": common.sha256_file(raw),
        }
    logits = artifact_root / "logits.f32"
    common.require(logits.is_file() and not logits.is_symlink(),
                   "llama logits file is unavailable")
    common.require(logits.stat().st_size == VOCABULARY_SIZE * 4,
                   "llama logits byte count differs")
    common.require(common.sha256_file(logits) == common.sha256_file(paths["result_output"]),
                   "llama logits and result_output differ")
    paths["logits"] = logits

    extractor_binary = extractor_root / "llama_logits_dump"
    common.require(extractor_binary.is_file() and not extractor_binary.is_symlink(),
                   "llama extractor binary is unavailable")
    binary_sha = common.sha256_file(extractor_binary)
    binary_record = common.read_optional_text(extractor_root, "binary-sha256.txt")
    source_record = common.read_optional_text(extractor_root, "source-sha256.txt")
    common.require(isinstance(binary_record, str)
                   and binary_record.split(maxsplit=1)[0] == binary_sha,
                   "llama extractor binary provenance differs")
    common.require(isinstance(source_record, str)
                   and source_record.split(maxsplit=1)[0] == EXTRACTOR_SOURCE_SHA256,
                   "llama extractor source provenance differs")
    return paths, {
        "artifact_root": str(artifact_root.resolve()),
        "manifest_sha256": common.sha256_file(manifest_path),
        "logits_sha256": common.sha256_file(logits),
        "checkpoints": checkpoint_provenance,
        "extractor_artifact_root": str(extractor_root.resolve()),
        "extractor_binary_sha256": binary_sha,
        "extractor_source_commit": EXTRACTOR_SOURCE_COMMIT,
        "extractor_source_sha256": EXTRACTOR_SOURCE_SHA256,
    }


def rms_norm(np: Any, value: Any, weight: Any) -> Any:
    inverse = np.reciprocal(
        np.sqrt(
            np.mean(value * value, axis=-1, dtype=np.float32)
            + np.float32(RMS_EPSILON),
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    return value * inverse[..., None] * weight


def silu(np: Any, value: Any) -> Any:
    return value * common.stable_sigmoid(np, value)


def linear_attention_residual(
    np: Any,
    layer_input: Any,
    layer: int,
    weight: Callable[[str], Any],
) -> Any:
    prefix = f"blk.{layer}"
    normalized = rms_norm(np, layer_input, weight(f"{prefix}.attn_norm.weight"))
    qkv_weight = weight(f"{prefix}.attn_qkv.weight")
    qkv = normalized @ qkv_weight.T
    del qkv_weight
    gate_weight = weight(f"{prefix}.attn_gate.weight")
    z = normalized @ gate_weight.T
    del gate_weight
    beta_weight = weight(f"{prefix}.ssm_beta.weight")
    beta_raw = normalized @ beta_weight.T
    del beta_weight
    alpha_weight = weight(f"{prefix}.ssm_alpha.weight")
    alpha_raw = normalized @ alpha_weight.T
    del alpha_weight

    conv_weight = weight(f"{prefix}.ssm_conv1d.weight")
    padded = np.zeros((layer_input.shape[0] + CONV_KERNEL - 1, QKV_SIZE),
                      dtype=np.float32)
    padded[CONV_KERNEL - 1:] = qkv
    conv_raw = np.empty_like(qkv, dtype=np.float32)
    for token in range(layer_input.shape[0]):
        conv_raw[token] = np.sum(
            padded[token:token + CONV_KERNEL] * conv_weight.T,
            axis=0,
            dtype=np.float32,
        )
    del conv_weight, padded, qkv
    conv_output = silu(np, conv_raw)
    del conv_raw

    tokens = layer_input.shape[0]
    query = conv_output[:, :QK_SIZE].reshape(tokens, KEY_HEADS, HEAD_DIM).copy()
    key = conv_output[:, QK_SIZE:2 * QK_SIZE].reshape(
        tokens, KEY_HEADS, HEAD_DIM
    ).copy()
    value = conv_output[:, 2 * QK_SIZE:].reshape(
        tokens, VALUE_HEADS, HEAD_DIM
    ).copy()
    del conv_output
    query *= np.reciprocal(
        np.sqrt(np.sum(query * query, axis=2, dtype=np.float32)
                + np.float32(RMS_EPSILON), dtype=np.float32),
        dtype=np.float32,
    )[:, :, None]
    key *= np.reciprocal(
        np.sqrt(np.sum(key * key, axis=2, dtype=np.float32)
                + np.float32(RMS_EPSILON), dtype=np.float32),
        dtype=np.float32,
    )[:, :, None]
    log_decay = weight(f"{prefix}.ssm_a")
    time_bias = weight(f"{prefix}.ssm_dt.bias")
    decay = log_decay[None, :] * common.stable_softplus(
        np, alpha_raw + time_bias[None, :]
    )
    beta = common.stable_sigmoid(np, beta_raw)
    del log_decay, time_bias, alpha_raw, beta_raw

    state = np.zeros((VALUE_HEADS, HEAD_DIM, HEAD_DIM), dtype=np.float32)
    recurrent = np.empty((tokens, VALUE_HEADS, HEAD_DIM), dtype=np.float32)
    head_map = np.arange(VALUE_HEADS) % KEY_HEADS
    scale = np.float32(1.0 / math.sqrt(HEAD_DIM))
    for token in range(tokens):
        token_query = query[token, head_map]
        token_key = key[token, head_map]
        state *= np.exp(decay[token], dtype=np.float32)[:, None, None]
        predicted = np.einsum(
            "hvk,hk->hv", state, token_key, dtype=np.float32, optimize=False
        )
        delta = (value[token] - predicted) * beta[token, :, None]
        state += delta[:, :, None] * token_key[:, None, :]
        recurrent[token] = np.einsum(
            "hvk,hk->hv", state, token_query, dtype=np.float32, optimize=False
        ) * scale
    del query, key, value, decay, beta, state
    recurrent_weight = weight(f"{prefix}.ssm_norm.weight")
    recurrent_inverse = np.reciprocal(
        np.sqrt(np.mean(recurrent * recurrent, axis=2, dtype=np.float32)
                + np.float32(RMS_EPSILON), dtype=np.float32),
        dtype=np.float32,
    )
    gated = (
        recurrent
        * recurrent_inverse[:, :, None]
        * recurrent_weight[None, None, :]
        * silu(np, z).reshape(tokens, VALUE_HEADS, HEAD_DIM)
    ).reshape(tokens, VALUE_SIZE)
    del recurrent, recurrent_inverse, recurrent_weight, z
    output_weight = weight(f"{prefix}.ssm_out.weight")
    residual = layer_input + gated @ output_weight.T
    del output_weight, gated, normalized
    common.require(residual.dtype == np.float32
                   and bool(np.isfinite(residual).all()),
                   f"layer {layer} linear attention residual is invalid")
    return residual


def full_attention_residual(
    np: Any,
    layer_input: Any,
    layer: int,
    weight: Callable[[str], Any],
) -> Any:
    prefix = f"blk.{layer}"
    mapped = {
        "blk.3.attn_norm.weight": weight(f"{prefix}.attn_norm.weight"),
        "blk.3.attn_q.weight": weight(f"{prefix}.attn_q.weight"),
        "blk.3.attn_k.weight": weight(f"{prefix}.attn_k.weight"),
        "blk.3.attn_v.weight": weight(f"{prefix}.attn_v.weight"),
        "blk.3.attn_q_norm.weight": weight(f"{prefix}.attn_q_norm.weight"),
        "blk.3.attn_k_norm.weight": weight(f"{prefix}.attn_k_norm.weight"),
        "blk.3.attn_output.weight": weight(f"{prefix}.attn_output.weight"),
    }
    residual, _ = full_attention.execute_full_attention(np, layer_input, mapped)
    del mapped
    return residual


def finish_dense_layer(
    np: Any,
    attention_residual: Any,
    layer: int,
    weight: Callable[[str], Any],
) -> Any:
    prefix = f"blk.{layer}"
    normalized = rms_norm(
        np, attention_residual, weight(f"{prefix}.post_attention_norm.weight")
    )
    gate_weight = weight(f"{prefix}.ffn_gate.weight")
    gate = normalized @ gate_weight.T
    del gate_weight
    up_weight = weight(f"{prefix}.ffn_up.weight")
    up = normalized @ up_weight.T
    del up_weight, normalized
    fused = silu(np, gate) * up
    del gate, up
    down_weight = weight(f"{prefix}.ffn_down.weight")
    output = attention_residual + fused @ down_weight.T
    del down_weight, fused
    common.require(output.dtype == np.float32 and bool(np.isfinite(output).all()),
                   f"layer {layer} output is invalid")
    return output


def tensor_sha256(np: Any, value: Any) -> str:
    return hashlib.sha256(
        np.asarray(value, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def measured(
    np: Any,
    actual: Any,
    expected: Any,
    *,
    actual_dtype: str,
    shape: list[int],
) -> dict[str, Any]:
    metrics = common.measure(np, actual, expected)
    metrics.update(
        {
            "shape": shape,
            "actual_logical_dtype": actual_dtype,
            "oracle_precision": "fp32",
            "actual_f32_sha256": tensor_sha256(np, actual),
            "expected_f32_sha256": tensor_sha256(np, expected),
        }
    )
    return metrics


def top_tokens(np: Any, logits: Any) -> dict[str, Any]:
    flat = np.asarray(logits, dtype=np.float32).reshape(-1)
    indices = np.argpartition(flat, -2)[-2:]
    indices = indices[np.argsort(flat[indices])[::-1]]
    return {
        "argmax_token_id": int(indices[0]),
        "runner_up_token_id": int(indices[1]),
        "top_logit": float(flat[indices[0]]),
        "runner_up_logit": float(flat[indices[1]]),
        "top2_margin": float(flat[indices[0]] - flat[indices[1]]),
    }


def build_reference(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    ferrum_source = common.git_provenance(repo_root, allow_untracked=True)
    llama_root = Path(args.llama_cpp_root).expanduser().resolve()
    common.require((llama_root / "gguf-py/gguf").is_dir(),
                   "--llama-cpp-root does not contain gguf-py/gguf")
    llama_source = common.git_provenance(llama_root, allow_untracked=False)
    sys.path.insert(0, str(llama_root / "gguf-py"))
    try:
        import numpy as np  # type: ignore
        import gguf  # type: ignore
        import yaml  # type: ignore
        from gguf import GGUFReader  # type: ignore
        from gguf.quants import dequantize  # type: ignore
    except ImportError as error:
        raise common.ReferenceError(
            "numpy and llama.cpp gguf-py are required; run with "
            "`uv run --with numpy --with pyyaml`"
        ) from error

    model_input = Path(args.model).expanduser().absolute()
    model_path = model_input.resolve()
    token_path = Path(args.prompt_token_ids).expanduser().resolve()
    capture_dir = Path(args.ferrum_capture).expanduser().resolve()
    llama_artifact = Path(args.llama_artifact).expanduser().resolve()
    extractor_artifact = Path(args.llama_extractor_artifact).expanduser().resolve()
    common.require(model_path.is_file() and not model_path.is_symlink(),
                   "--model must resolve to a regular GGUF file")
    common.require(token_path.is_file() and not token_path.is_symlink(),
                   "--prompt-token-ids must be a regular JSON file")
    common.require(capture_dir.is_dir(), "--ferrum-capture must be a directory")
    common.require(llama_artifact.is_dir(), "--llama-artifact must be a directory")
    common.require(extractor_artifact.is_dir(),
                   "--llama-extractor-artifact must be a directory")
    token_ids = common.load_token_ids(token_path)
    token_count = len(token_ids)
    common.require(token_count <= full_attention.TOKENS_MAXIMUM,
                   "full-model fixture exceeds token maximum")
    actual_paths, actual_provenance = load_ferrum_capture(capture_dir, token_count)
    llama_paths, llama_provenance = load_llama_artifact(
        llama_artifact, extractor_artifact, token_count
    )

    reader = GGUFReader(str(model_path), "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    common.require(len(tensors) == len(reader.tensors) == 426,
                   "GGUF tensor inventory differs from the reviewed model")
    metadata: dict[str, Any] = {}
    for key, expected in common.EXPECTED_METADATA.items():
        field = reader.get_field(key)
        common.require(field is not None, f"GGUF is missing metadata: {key}")
        actual = common.json_value(field.contents())
        if isinstance(expected, float):
            common.require(math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12),
                           f"GGUF metadata {key} differs")
        else:
            common.require(actual == expected, f"GGUF metadata {key} differs")
        metadata[key] = actual
        common.require(common.json_value(reader.get_field("qwen35.feed_forward_length").contents())
                   == INTERMEDIATE_SIZE, "GGUF feed-forward size differs")
    for key in sorted(reader.fields):
        if key.startswith(("general.", "qwen35.", "quantize.")):
            metadata[key] = common.json_value(reader.fields[key].contents())
    common.write_json(out_dir / "gguf-metadata.json", metadata)
    inventory = [
        {
            "name": tensor.name,
            "shape": [int(value) for value in tensor.shape],
            "tensor_type": tensor.tensor_type.name,
            "element_count": int(tensor.n_elements),
            "byte_count": int(tensor.n_bytes),
        }
        for tensor in sorted(reader.tensors, key=lambda item: item.name)
    ]
    common.write_json(out_dir / "tensor-inventory.json", inventory)

    required = {"token_embd.weight", "output_norm.weight"}
    for layer in range(LAYER_COUNT):
        prefix = f"blk.{layer}"
        required.update(
            {
                f"{prefix}.attn_norm.weight",
                f"{prefix}.post_attention_norm.weight",
                f"{prefix}.ffn_gate.weight",
                f"{prefix}.ffn_up.weight",
                f"{prefix}.ffn_down.weight",
            }
        )
        if (layer + 1) % FULL_ATTENTION_INTERVAL == 0:
            required.update(
                f"{prefix}.{suffix}"
                for suffix in (
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_q_norm.weight",
                    "attn_k_norm.weight",
                    "attn_output.weight",
                )
            )
        else:
            required.update(
                f"{prefix}.{suffix}"
                for suffix in (
                    "attn_qkv.weight",
                    "attn_gate.weight",
                    "ssm_beta.weight",
                    "ssm_alpha.weight",
                    "ssm_conv1d.weight",
                    "ssm_a",
                    "ssm_dt.bias",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                )
            )
    common.require(not required - set(tensors),
                   f"GGUF is missing required tensors: {sorted(required - set(tensors))}")

    def weight(name: str) -> Any:
        tensor = tensors[name]
        value = dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False)
        common.require(bool(np.isfinite(value).all()),
                       f"dequantized tensor is non-finite: {name}")
        return value

    reference_dir = out_dir / "layers"
    reference_dir.mkdir()
    token_index = np.asarray(token_ids, dtype=np.int64)
    embedding_tensor = tensors["token_embd.weight"]
    hidden = dequantize(
        embedding_tensor.data[token_index].copy(), embedding_tensor.tensor_type
    ).astype(np.float32, copy=False)
    common.require(hidden.shape == (token_count, HIDDEN_SIZE),
                   "reference embedding shape differs")
    embedding_path = out_dir / "embedding.f32"
    np.asarray(hidden, dtype="<f4").tofile(embedding_path)
    actual_embedding = np.fromfile(
        actual_paths[EMBEDDING_VALUE_ID], dtype="<f2"
    ).astype(np.float32).reshape(token_count, HIDDEN_SIZE)
    llama_embedding = np.fromfile(
        llama_paths["model.input_embed"], dtype="<f4"
    ).reshape(token_count, HIDDEN_SIZE)
    embedding_metrics = measured(
        np, actual_embedding, hidden,
        actual_dtype="fp16", shape=[token_count, HIDDEN_SIZE]
    )
    layer_records: list[dict[str, Any]] = []
    llama_layer_metrics: list[dict[str, Any]] = []
    for layer in range(LAYER_COUNT):
        layer_kind = (
            "full_attention"
            if (layer + 1) % FULL_ATTENTION_INTERVAL == 0
            else "linear_attention"
        )
        if layer_kind == "full_attention":
            attention_residual = full_attention_residual(np, hidden, layer, weight)
        else:
            attention_residual = linear_attention_residual(np, hidden, layer, weight)
        next_hidden = finish_dense_layer(np, attention_residual, layer, weight)
        del attention_residual, hidden
        hidden = next_hidden
        path = reference_dir / f"layer-{layer:02d}-output.f32"
        np.asarray(hidden, dtype="<f4").tofile(path)
        actual = np.fromfile(
            actual_paths[layer_value_id(layer)], dtype="<f2"
        ).astype(np.float32).reshape(token_count, HIDDEN_SIZE)
        metrics = measured(
            np, actual, hidden,
            actual_dtype="fp16", shape=[token_count, HIDDEN_SIZE]
        )
        layer_records.append(
            {
                "layer_index": layer,
                "layer_kind": layer_kind,
                "raw_file": str(path.relative_to(out_dir)),
                "raw_sha256": common.sha256_file(path),
                "actual_raw_sha256": actual_provenance["checkpoints"][
                    layer_value_id(layer)
                ]["raw_sha256"],
                "metrics": metrics,
            }
        )
        llama = np.fromfile(llama_paths[f"l_out-{layer}"], dtype="<f4").reshape(
            -1, HIDDEN_SIZE
        )
        llama_expected = hidden[-llama.shape[0]:]
        llama_layer_metrics.append(
            {
                "layer_index": layer,
                "layer_kind": layer_kind,
                "metrics": measured(
                    np, llama, llama_expected,
                    actual_dtype="fp32",
                    shape=list(llama.shape),
                ),
            }
        )
        print(f"Qwen3.5 FP32 oracle layer {layer + 1}/{LAYER_COUNT} complete",
              file=sys.stderr, flush=True)
        gc.collect()

    final_norm_weight = weight("output_norm.weight")
    final_hidden = rms_norm(np, hidden, final_norm_weight)
    del final_norm_weight, hidden
    final_path = out_dir / "final-hidden.f32"
    np.asarray(final_hidden, dtype="<f4").tofile(final_path)
    actual_final = np.fromfile(
        actual_paths[FINAL_HIDDEN_VALUE_ID], dtype="<f2"
    ).astype(np.float32).reshape(token_count, HIDDEN_SIZE)
    final_metrics = measured(
        np, actual_final, final_hidden,
        actual_dtype="fp16", shape=[token_count, HIDDEN_SIZE]
    )
    llama_final = np.fromfile(llama_paths["result_norm"], dtype="<f4").reshape(
        1, HIDDEN_SIZE
    )
    llama_final_metrics = measured(
        np, llama_final, final_hidden[-1:],
        actual_dtype="fp32", shape=[1, HIDDEN_SIZE]
    )

    reference_logits = np.empty(VOCABULARY_SIZE, dtype=np.float32)
    isolated_logits = np.empty(VOCABULARY_SIZE, dtype=np.float32)
    for start in range(0, VOCABULARY_SIZE, LOGIT_CHUNK_ROWS):
        end = min(VOCABULARY_SIZE, start + LOGIT_CHUNK_ROWS)
        rows = dequantize(
            embedding_tensor.data[start:end].copy(), embedding_tensor.tensor_type
        ).astype(np.float32, copy=False)
        reference_logits[start:end] = rows @ final_hidden[-1]
        isolated_logits[start:end] = rows @ actual_final[-1]
        del rows
    logits_path = out_dir / "full-vocabulary-logits.f32"
    isolated_path = out_dir / "ferrum-final-hidden-head-reference.f32"
    np.asarray(reference_logits, dtype="<f4").tofile(logits_path)
    np.asarray(isolated_logits, dtype="<f4").tofile(isolated_path)
    actual_logits = np.fromfile(
        actual_paths[LOGITS_VALUE_ID], dtype="<f2"
    ).astype(np.float32).reshape(VOCABULARY_SIZE)
    llama_logits = np.fromfile(llama_paths["logits"], dtype="<f4").reshape(
        VOCABULARY_SIZE
    )
    logits_metrics = measured(
        np, actual_logits, reference_logits,
        actual_dtype="fp16", shape=[VOCABULARY_SIZE]
    )
    isolated_head_metrics = measured(
        np, actual_logits, isolated_logits,
        actual_dtype="fp16", shape=[VOCABULARY_SIZE]
    )
    llama_logits_metrics = measured(
        np, llama_logits, reference_logits,
        actual_dtype="fp32", shape=[VOCABULARY_SIZE]
    )

    metadata_path = out_dir / "gguf-metadata.json"
    inventory_path = out_dir / "tensor-inventory.json"
    source_path = Path(__file__).resolve()
    report = {
        "schema_version": 1,
        "status": "measured",
        "oracle": {
            "identity": "cpu.fp32.python.qwen35_gguf_model_reference",
            "precision": "fp32",
            "semantics": (
                "independent streamed Qwen3.5 dense-hybrid transformer stack and "
                "tied full-vocabulary head over GGUF-dequantized weights"
            ),
            "source_path": str(source_path.relative_to(repo_root)),
            "source_sha256": common.sha256_file(source_path),
            "linear_common_source_path": str(Path(common.__file__).resolve().relative_to(repo_root)),
            "linear_common_source_sha256": common.sha256_file(Path(common.__file__).resolve()),
            "full_attention_common_source_path": str(
                Path(full_attention.__file__).resolve().relative_to(repo_root)
            ),
            "full_attention_common_source_sha256": common.sha256_file(
                Path(full_attention.__file__).resolve()
            ),
            "ferrum_source": ferrum_source,
            "llama_cpp_gguf_py_source": llama_source,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pyyaml_version": yaml.__version__,
            "gguf_module_path": str(Path(gguf.__file__).resolve()),
        },
        "model": {
            "path": str(model_path),
            "sha256": common.sha256_file(model_path),
            "byte_count": model_path.stat().st_size,
            "format": "GGUF Q4_K_M",
            "hugging_face_snapshot": common.hf_snapshot_identity(model_input),
            "upstream_model": metadata.get("general.base_model.0.repo_url"),
            "upstream_revision": "not_declared_in_gguf",
            "converter_version": "not_declared_in_gguf",
            "quantized_by": metadata.get("general.quantized_by"),
            "quantization_version": metadata["general.quantization_version"],
            "license": metadata.get("general.license"),
            "metadata_file": metadata_path.name,
            "metadata_sha256": common.sha256_file(metadata_path),
            "tensor_inventory_file": inventory_path.name,
            "tensor_inventory_sha256": common.sha256_file(inventory_path),
            "tensor_count": len(inventory),
        },
        "fixture": {
            "fixture_id": f"qwen35-4b.gguf-q4-k-m.full-model.tokens-{token_count}",
            "token_ids": token_ids,
            "token_sequence_sha256": hashlib.sha256(
                b"".join(int(value).to_bytes(4, "little") for value in token_ids)
            ).hexdigest(),
            "tokens": token_count,
            "hidden_size": HIDDEN_SIZE,
            "layers": LAYER_COUNT,
            "full_attention_interval": FULL_ATTENTION_INTERVAL,
            "vocabulary_size": VOCABULARY_SIZE,
            "quantized_embedding_is_model_input": True,
            "tied_embedding_lm_head": True,
        },
        "actual": actual_provenance,
        "reference": {
            "embedding": {
                "raw_file": embedding_path.name,
                "raw_sha256": common.sha256_file(embedding_path),
                "logical_dtype": "fp32",
                "logical_shape": [token_count, HIDDEN_SIZE],
                "metrics": embedding_metrics,
            },
            "layers": layer_records,
            "final_hidden": {
                "raw_file": final_path.name,
                "raw_sha256": common.sha256_file(final_path),
                "logical_dtype": "fp32",
                "logical_shape": [token_count, HIDDEN_SIZE],
                "metrics": final_metrics,
            },
            "full_vocabulary_logits": {
                "raw_file": logits_path.name,
                "raw_sha256": common.sha256_file(logits_path),
                "logical_dtype": "fp32",
                "logical_shape": [VOCABULARY_SIZE],
                "metrics": logits_metrics,
                "top_tokens": top_tokens(np, reference_logits),
            },
            "isolated_ferrum_final_hidden_head": {
                "raw_file": isolated_path.name,
                "raw_sha256": common.sha256_file(isolated_path),
                "logical_dtype": "fp32",
                "logical_shape": [VOCABULARY_SIZE],
                "metrics": isolated_head_metrics,
                "top_tokens": top_tokens(np, isolated_logits),
            },
            "actual_top_tokens": top_tokens(np, actual_logits),
        },
        "llama_cpp_cross_validation": {
            "role": "same-GGUF quantized implementation cross-validation only",
            "provenance": llama_provenance,
            "embedding_metrics": measured(
                np, llama_embedding, np.fromfile(embedding_path, dtype="<f4").reshape(
                    token_count, HIDDEN_SIZE
                ),
                actual_dtype="fp32", shape=[token_count, HIDDEN_SIZE]
            ),
            "layer_metrics": llama_layer_metrics,
            "final_hidden_metrics": llama_final_metrics,
            "full_vocabulary_logits_metrics": llama_logits_metrics,
            "top_tokens": top_tokens(np, llama_logits),
        },
        "invocation": {"argv": [str(value) for value in sys.argv], "cwd": os.getcwd()},
    }
    common.write_json(out_dir / "report.json", report)
    return report


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="qwen35-model-reference-") as tmp:
        root = Path(tmp)
        capture = root / "capture"
        capture.mkdir()
        identity = {
            "plan_id": "plan-1",
            "plan_hash": "a" * 64,
            "model_id": "Qwen3.5-4B-Q4_K_M",
            "family_fingerprint": "b" * 64,
            "program_fingerprint": "c" * 64,
            "run_id": "run-1",
        }
        common.write_json(capture / "plan.json", {"schema_version": 1, **identity})
        value_ids = [EMBEDDING_VALUE_ID]
        value_ids.extend(layer_value_id(layer) for layer in range(LAYER_COUNT))
        value_ids.extend((FINAL_HIDDEN_VALUE_ID, LOGITS_VALUE_ID))
        records = []
        for index, value_id in enumerate(value_ids):
            elements = VOCABULARY_SIZE if value_id == LOGITS_VALUE_ID else HIDDEN_SIZE
            dimensions = [1, VOCABULARY_SIZE] if value_id == LOGITS_VALUE_ID else [128, HIDDEN_SIZE]
            raw_name = f"checkpoint-{index}.bin"
            payload = struct.pack("<e", float(index + 1)) * elements
            (capture / raw_name).write_bytes(payload)
            records.append(
                {
                    "value": {
                        "value_id": value_id,
                        "tensor": {"dimensions": dimensions, "element_type": "f16"},
                    },
                    "participant_index": 0,
                    "token_span": {
                        "immediate_tokens": 1,
                        "full_input_tokens": 1,
                        "immediate_start_token": 0,
                        "immediate_end_token": 1,
                    },
                    "output_layout": {"element_type": "f16", "element_count": elements},
                    "raw_file": raw_name,
                    "raw_bytes": len(payload),
                    "raw_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        common.write_json(
            capture / "wave-0000.json",
            {
                "schema_version": 1,
                **identity,
                "wave_kind": "prefill",
                "participant_count": 1,
                "records": records,
            },
        )
        paths, provenance = load_ferrum_capture(capture, 1)
        common.require(len(paths) == LAYER_COUNT + 3,
                       "synthetic model capture did not roundtrip")
        common.require(len(provenance["checkpoints"]) == LAYER_COUNT + 3,
                       "synthetic model provenance is incomplete")
    print(SELF_TEST_PASS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--prompt-token-ids")
    parser.add_argument("--ferrum-capture")
    parser.add_argument("--llama-cpp-root")
    parser.add_argument("--llama-artifact")
    parser.add_argument("--llama-extractor-artifact")
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
        "--llama-artifact": args.llama_artifact,
        "--llama-extractor-artifact": args.llama_extractor_artifact,
        "--out": args.out,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        print(f"missing required arguments: {', '.join(missing)}", file=sys.stderr)
        return 2
    out_dir = Path(args.out).expanduser().resolve()
    try:
        common.require(not out_dir.exists() or out_dir.is_dir(),
                       f"output path is not a directory: {out_dir}")
        common.require(not out_dir.exists() or not any(out_dir.iterdir()),
                       f"output directory is not empty: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        build_reference(args, out_dir)
    except Exception as error:
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            common.write_json(
                out_dir / "failure.json",
                {
                    "schema_version": 1,
                    "status": "fail",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        except OSError:
            pass
        print(f"QWEN35 GGUF MODEL REFERENCE FAIL: {error}", file=sys.stderr)
        return 1
    print(f"{PASS_PREFIX}: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
