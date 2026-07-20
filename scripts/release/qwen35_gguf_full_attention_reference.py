#!/usr/bin/env python3
"""Build an independent FP32 Qwen3.5 layer-3 full-attention reference."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

import qwen35_gguf_linear_attention_reference as common


PASS_PREFIX = "QWEN35 GGUF FULL ATTENTION REFERENCE PASS"
SELF_TEST_PASS = "QWEN35 GGUF FULL ATTENTION REFERENCE SELF-TEST PASS"
INPUT_VALUE_ID = "value.layer.2.output"
OUTPUT_VALUE_ID = "value.layer.3.attention"
TOKENS_MAXIMUM = 128
HIDDEN_SIZE = 2560
QUERY_HEADS = 16
KEY_VALUE_HEADS = 4
HEAD_DIM = 256
ROPE_DIM = 64
QUERY_SIZE = QUERY_HEADS * HEAD_DIM
QUERY_PROJECTION_SIZE = QUERY_SIZE * 2
KEY_VALUE_SIZE = KEY_VALUE_HEADS * HEAD_DIM
ROPE_THETA = 10_000_000.0
RMS_EPSILON = 1.0e-6

EXPECTED_TENSORS: dict[str, tuple[tuple[int, ...], str]] = {
    "blk.3.attn_norm.weight": ((HIDDEN_SIZE,), "F32"),
    "blk.3.attn_q.weight": ((HIDDEN_SIZE, QUERY_PROJECTION_SIZE), "Q4_K"),
    "blk.3.attn_k.weight": ((HIDDEN_SIZE, KEY_VALUE_SIZE), "Q4_K"),
    "blk.3.attn_v.weight": ((HIDDEN_SIZE, KEY_VALUE_SIZE), "Q6_K"),
    "blk.3.attn_q_norm.weight": ((HEAD_DIM,), "F32"),
    "blk.3.attn_k_norm.weight": ((HEAD_DIM,), "F32"),
    "blk.3.attn_output.weight": ((QUERY_SIZE, HIDDEN_SIZE), "Q4_K"),
}


def load_capture(
    capture_dir: Path, token_count: int
) -> tuple[dict[str, Path], dict[str, Any]]:
    plan_path = capture_dir / "plan.json"
    wave_paths = sorted(capture_dir.glob("wave-*.json"))
    common.require(plan_path.is_file(), f"missing Ferrum capture plan: {plan_path}")
    common.require(len(wave_paths) == 1, "expected exactly one Ferrum capture wave")
    plan = common.load_json(plan_path)
    wave = common.load_json(wave_paths[0])
    common.require(isinstance(plan, dict) and isinstance(wave, dict),
                   "Ferrum plan and wave must be objects")
    common.require(plan.get("schema_version") == 1 and wave.get("schema_version") == 1,
                   "unsupported Ferrum capture schema")
    common.require(wave.get("wave_kind") == "prefill"
                   and wave.get("participant_count") == 1,
                   "Ferrum checkpoint must be a single-participant prefill")
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
    common.require(isinstance(records, list), "Ferrum wave records must be a list")

    paths: dict[str, Path] = {}
    checkpoint_provenance: dict[str, Any] = {}
    for value_id in (INPUT_VALUE_ID, OUTPUT_VALUE_ID):
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
        common.require(isinstance(span, dict), f"{value_id} token_span must be an object")
        common.require(
            span.get("immediate_tokens") == token_count
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
        dimensions = tensor.get("dimensions")
        common.require(isinstance(dimensions, list) and len(dimensions) == 2,
                       f"{value_id} must have rank-2 capacity")
        common.require(dimensions[0] >= token_count and dimensions[1] == HIDDEN_SIZE,
                       f"{value_id} capacity differs from the fixture")
        elements = token_count * HIDDEN_SIZE
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
        checkpoint_provenance[value_id] = {
            "logical_dtype": "fp16",
            "logical_shape": [token_count, HIDDEN_SIZE],
            "raw_file": raw_path.name,
            "raw_sha256": raw_sha,
        }

    artifact_root = capture_dir.parent
    provenance = {
        "artifact_root": str(artifact_root.resolve()),
        "plan_file": plan_path.name,
        "plan_sha256": common.sha256_file(plan_path),
        "wave_file": wave_paths[0].name,
        "wave_sha256": common.sha256_file(wave_paths[0]),
        "git_sha": common.read_optional_text(artifact_root, "git-sha.txt"),
        "git_status": common.read_optional_text(artifact_root, "git-status.txt"),
        "tracked_diff_sha256": (
            common.sha256_file(artifact_root / "tracked-diff.patch")
            if (artifact_root / "tracked-diff.patch").is_file()
            else None
        ),
        "binary_sha256_record": common.read_optional_text(
            artifact_root, "binary-sha256.txt"
        ),
        "checkpoints": checkpoint_provenance,
    }
    return paths, provenance


def load_llama_checkpoints(
    checkpoint_dir: Path, token_count: int
) -> tuple[dict[str, Path], dict[str, Any]]:
    manifest_path = checkpoint_dir / "checkpoint-manifest.json"
    manifest = common.load_json(manifest_path)
    common.require(isinstance(manifest, dict) and manifest.get("schema_version") == 1,
                   "unsupported llama checkpoint manifest")
    common.require(manifest.get("status") == "pass"
                   and manifest.get("output_dtype") == "f32",
                   "llama checkpoint manifest is not successful FP32")
    records = manifest.get("records")
    common.require(isinstance(records, list), "llama checkpoint records must be a list")
    paths: dict[str, Path] = {}
    provenance: dict[str, Any] = {
        "artifact_root": str(checkpoint_dir.parent.resolve()),
        "manifest_sha256": common.sha256_file(manifest_path),
        "checkpoints": {},
    }
    for name in ("l_out-2", "attn_residual-3"):
        selected = [
            record for record in records
            if isinstance(record, dict) and record.get("tensor_name") == name
        ]
        common.require(len(selected) == 1, f"llama checkpoint must contain one {name}")
        record = selected[0]
        expected_shape = [token_count, HIDDEN_SIZE]
        common.require(record.get("logical_shape") == expected_shape
                       and record.get("element_count") == token_count * HIDDEN_SIZE,
                       f"llama {name} shape differs from the fixture")
        raw_path = common.safe_child(checkpoint_dir, record.get("raw_file"),
                                     f"llama.{name}.raw_file")
        common.require(raw_path.stat().st_size == token_count * HIDDEN_SIZE * 4,
                       f"llama {name} byte count differs")
        paths[name] = raw_path
        provenance["checkpoints"][name] = {
            "raw_file": raw_path.name,
            "raw_sha256": common.sha256_file(raw_path),
            "logical_dtype": "fp32",
            "logical_shape": expected_shape,
        }
    for name in (
        "extractor-binary-sha256.txt",
        "extractor-source-sha256.txt",
        "llama-runtime-version.txt",
    ):
        value = common.read_optional_text(checkpoint_dir.parent, name)
        common.require(value is not None, f"llama artifact is missing {name}")
        provenance[name.removesuffix(".txt")] = value
    return paths, provenance


def tensor_f32_sha256(np: Any, value: Any) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f4").tobytes(order="C")).hexdigest()


def execute_full_attention(
    np: Any,
    layer_input: Any,
    weights: dict[str, Any],
) -> tuple[Any, dict[str, str]]:
    def rms_norm(value: Any, weight: Any) -> Any:
        inverse = np.reciprocal(
            np.sqrt(
                np.mean(value * value, axis=-1, dtype=np.float32)
                + np.float32(RMS_EPSILON),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        return value * inverse[..., None] * weight

    normalized = rms_norm(layer_input, weights["blk.3.attn_norm.weight"])
    query_raw = normalized @ weights["blk.3.attn_q.weight"].T
    key_raw = normalized @ weights["blk.3.attn_k.weight"].T
    value_raw = normalized @ weights["blk.3.attn_v.weight"].T
    token_count = layer_input.shape[0]
    query_gate = query_raw.reshape(token_count, QUERY_HEADS, 2, HEAD_DIM)
    query = query_gate[:, :, 0, :].copy()
    gate = query_gate[:, :, 1, :].copy()
    key = key_raw.reshape(token_count, KEY_VALUE_HEADS, HEAD_DIM).copy()
    value = value_raw.reshape(token_count, KEY_VALUE_HEADS, HEAD_DIM).copy()
    query = rms_norm(query, weights["blk.3.attn_q_norm.weight"])
    key = rms_norm(key, weights["blk.3.attn_k_norm.weight"])

    half_rope = ROPE_DIM // 2
    exponent = (
        -np.float32(2.0) * np.arange(half_rope, dtype=np.float32)
        / np.float32(ROPE_DIM)
    )
    frequency = np.power(np.float32(ROPE_THETA), exponent, dtype=np.float32)
    angles = np.arange(token_count, dtype=np.float32)[:, None] * frequency[None, :]
    cosine = np.cos(angles, dtype=np.float32)[:, None, :]
    sine = np.sin(angles, dtype=np.float32)[:, None, :]
    for values in (query, key):
        low = values[:, :, :half_rope].copy()
        high = values[:, :, half_rope:ROPE_DIM].copy()
        values[:, :, :half_rope] = low * cosine - high * sine
        values[:, :, half_rope:ROPE_DIM] = low * sine + high * cosine

    context = np.empty((token_count, QUERY_HEADS, HEAD_DIM), dtype=np.float32)
    repeat = QUERY_HEADS // KEY_VALUE_HEADS
    scale = np.float32(1.0 / math.sqrt(HEAD_DIM))
    for token in range(token_count):
        for query_head in range(QUERY_HEADS):
            key_value_head = query_head // repeat
            scores = (key[:token + 1, key_value_head] @ query[token, query_head]) * scale
            scores -= np.max(scores)
            probabilities = np.exp(scores, dtype=np.float32)
            probabilities /= np.sum(probabilities, dtype=np.float32)
            context[token, query_head] = probabilities @ value[:token + 1, key_value_head]
    context *= common.stable_sigmoid(np, gate)
    projected = (
        context.reshape(token_count, QUERY_SIZE)
        @ weights["blk.3.attn_output.weight"].T
    )
    residual = layer_input + projected
    common.require(residual.shape == (token_count, HIDDEN_SIZE),
                   "full-attention residual shape differs")
    common.require(residual.dtype == np.float32,
                   "full-attention residual is not FP32")
    common.require(bool(np.isfinite(residual).all()),
                   "full-attention residual contains NaN or Inf")
    stages = {
        "input_norm_f32_sha256": tensor_f32_sha256(np, normalized),
        "query_raw_f32_sha256": tensor_f32_sha256(np, query_raw),
        "key_raw_f32_sha256": tensor_f32_sha256(np, key_raw),
        "value_raw_f32_sha256": tensor_f32_sha256(np, value_raw),
        "query_rope_f32_sha256": tensor_f32_sha256(np, query),
        "key_rope_f32_sha256": tensor_f32_sha256(np, key),
        "context_gated_f32_sha256": tensor_f32_sha256(np, context),
        "projected_f32_sha256": tensor_f32_sha256(np, projected),
        "residual_f32_sha256": tensor_f32_sha256(np, residual),
    }
    return residual, stages


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

    model_input_path = Path(args.model).expanduser().absolute()
    model_path = model_input_path.resolve()
    token_path = Path(args.prompt_token_ids).expanduser().resolve()
    capture_dir = Path(args.ferrum_capture).expanduser().resolve()
    common.require(model_path.is_file() and not model_path.is_symlink(),
                   "--model must resolve to a regular GGUF file")
    common.require(token_path.is_file() and not token_path.is_symlink(),
                   "--prompt-token-ids must be a regular JSON file")
    common.require(capture_dir.is_dir(), "--ferrum-capture must be a directory")
    token_ids = common.load_token_ids(token_path)
    token_count = len(token_ids)
    common.require(token_count <= TOKENS_MAXIMUM, "reference fixture exceeds token maximum")
    capture_paths, actual_provenance = load_capture(capture_dir, token_count)

    reader = GGUFReader(str(model_path), "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    common.require(len(tensors) == len(reader.tensors),
                   "GGUF contains duplicate tensor names")
    metadata: dict[str, Any] = {}
    for key, expected in common.EXPECTED_METADATA.items():
        field = reader.get_field(key)
        common.require(field is not None, f"GGUF is missing metadata: {key}")
        actual = common.json_value(field.contents())
        if isinstance(expected, float):
            common.require(math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12),
                           f"GGUF metadata {key} differs: {actual} != {expected}")
        else:
            common.require(actual == expected,
                           f"GGUF metadata {key} differs: {actual} != {expected}")
        metadata[key] = actual
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
    for name, (expected_shape, expected_type) in EXPECTED_TENSORS.items():
        common.require(name in tensors, f"GGUF is missing tensor: {name}")
        tensor = tensors[name]
        shape = tuple(int(value) for value in tensor.shape)
        common.require(shape == expected_shape,
                       f"GGUF tensor {name} shape differs: {shape} != {expected_shape}")
        common.require(tensor.tensor_type.name == expected_type,
                       f"GGUF tensor {name} type differs: "
                       f"{tensor.tensor_type.name} != {expected_type}")

    weights: dict[str, Any] = {}
    for name in EXPECTED_TENSORS:
        tensor = tensors[name]
        value = dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False)
        common.require(bool(np.isfinite(value).all()),
                       f"dequantized tensor is non-finite: {name}")
        weights[name] = value

    layer_input = np.fromfile(capture_paths[INPUT_VALUE_ID], dtype="<f2").astype(
        np.float32
    ).reshape(token_count, HIDDEN_SIZE)
    actual_output = np.fromfile(capture_paths[OUTPUT_VALUE_ID], dtype="<f2").astype(
        np.float32
    ).reshape(token_count, HIDDEN_SIZE)
    reference, stage_hashes = execute_full_attention(np, layer_input, weights)
    reference_path = out_dir / "layer-3-full-attention-residual.f32"
    np.asarray(reference, dtype="<f4").tofile(reference_path)
    metrics = common.measure(np, actual_output, reference)
    metrics.update(
        {
            "shape": [token_count, HIDDEN_SIZE],
            "actual_logical_dtype": "fp16",
            "oracle_precision": "fp32",
            "actual_f32_sha256": tensor_f32_sha256(np, actual_output),
            "expected_f32_sha256": common.sha256_file(reference_path),
        }
    )

    cross_validation = None
    if args.llama_checkpoints:
        llama_dir = Path(args.llama_checkpoints).expanduser().resolve()
        llama_paths, llama_provenance = load_llama_checkpoints(llama_dir, token_count)
        llama_input = np.fromfile(llama_paths["l_out-2"], dtype="<f4").reshape(
            token_count, HIDDEN_SIZE
        )
        llama_output = np.fromfile(llama_paths["attn_residual-3"], dtype="<f4").reshape(
            token_count, HIDDEN_SIZE
        )
        llama_reference, llama_stage_hashes = execute_full_attention(np, llama_input, weights)
        cross_validation = {
            "role": "same-GGUF quantized implementation cross-validation only",
            "provenance": llama_provenance,
            "input_ferrum_vs_llama": common.measure(np, layer_input, llama_input),
            "residual_llama_vs_its_fp32_oracle": common.measure(
                np, llama_output, llama_reference
            ),
            "oracle_stage_f32_sha256": llama_stage_hashes,
        }

    metadata_path = out_dir / "gguf-metadata.json"
    inventory_path = out_dir / "tensor-inventory.json"
    prompt_sha = hashlib.sha256(
        b"".join(int(value).to_bytes(4, "little") for value in token_ids)
    ).hexdigest()
    source_path = Path(__file__).resolve()
    common_path = Path(common.__file__).resolve()
    report = {
        "schema_version": 1,
        "status": "measured",
        "oracle": {
            "identity": "cpu.fp32.python.qwen35_gguf_full_attention_reference",
            "precision": "fp32",
            "semantics": (
                "independent Qwen3.5 layer-3 RMSNorm, gated QKV, Q/K norm, "
                "partial RoPE, causal GQA, output projection, and residual"
            ),
            "source_path": str(source_path.relative_to(repo_root)),
            "source_sha256": common.sha256_file(source_path),
            "common_source_path": str(common_path.relative_to(repo_root)),
            "common_source_sha256": common.sha256_file(common_path),
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
            "hugging_face_snapshot": common.hf_snapshot_identity(model_input_path),
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
            "fixture_id": (
                "qwen35-4b.gguf-q4-k-m.layer-3.full-attention."
                f"tokens-{token_count}"
            ),
            "layer_index": 3,
            "input_value_id": INPUT_VALUE_ID,
            "output_value_id": OUTPUT_VALUE_ID,
            "token_ids": token_ids,
            "token_sequence_sha256": prompt_sha,
            "shape": [token_count, HIDDEN_SIZE],
            "query_heads": QUERY_HEADS,
            "key_value_heads": KEY_VALUE_HEADS,
            "head_dim": HEAD_DIM,
            "rope_dim": ROPE_DIM,
            "rope_theta": "10000000",
            "rope_interleaved": False,
            "output_gate": True,
            "causal": True,
        },
        "actual": actual_provenance,
        "reference": {
            "raw_file": reference_path.name,
            "raw_sha256": common.sha256_file(reference_path),
            "logical_dtype": "fp32",
            "logical_shape": [token_count, HIDDEN_SIZE],
            "stage_f32_sha256": stage_hashes,
        },
        "metrics": metrics,
        "llama_cpp_cross_validation": cross_validation,
        "invocation": {"argv": [str(value) for value in sys.argv], "cwd": os.getcwd()},
    }
    common.write_json(out_dir / "report.json", report)
    return report


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="qwen35-full-reference-") as tmp:
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
        records = []
        for index, value_id in enumerate((INPUT_VALUE_ID, OUTPUT_VALUE_ID)):
            raw_name = f"checkpoint-{index}.bin"
            payload = struct.pack("<e", float(index + 1)) * HIDDEN_SIZE
            (capture / raw_name).write_bytes(payload)
            records.append(
                {
                    "value": {
                        "value_id": value_id,
                        "tensor": {
                            "dimensions": [TOKENS_MAXIMUM, HIDDEN_SIZE],
                            "element_type": "f16",
                        },
                    },
                    "participant_index": 0,
                    "token_span": {
                        "immediate_tokens": 1,
                        "full_input_tokens": 1,
                        "immediate_start_token": 0,
                        "immediate_end_token": 1,
                    },
                    "output_layout": {
                        "element_type": "f16",
                        "element_count": HIDDEN_SIZE,
                    },
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
        paths, provenance = load_capture(capture, 1)
        common.require(set(paths) == {INPUT_VALUE_ID, OUTPUT_VALUE_ID},
                       "synthetic capture values did not roundtrip")
        common.require(len(provenance["checkpoints"]) == 2,
                       "synthetic capture provenance is incomplete")
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
        print(f"QWEN35 GGUF FULL ATTENTION REFERENCE FAIL: {error}", file=sys.stderr)
        return 1
    print(f"{PASS_PREFIX}: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
