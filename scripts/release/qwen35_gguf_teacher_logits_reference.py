#!/usr/bin/env python3
"""Build same-history CPU FP32 decision logits for Qwen3.5 GGUF."""

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
from typing import Any

import qwen35_gguf_linear_attention_reference as common
import qwen35_gguf_model_reference as model_reference


PASS_PREFIX = "QWEN35 GGUF TEACHER LOGITS REFERENCE PASS"
SELF_TEST_PASS = "QWEN35 GGUF TEACHER LOGITS REFERENCE SELF-TEST PASS"
PROMPT_FIELDS = frozenset(
    {
        "schema_version",
        "encoding",
        "request_id",
        "token_count",
        "token_ids_sha256",
        "token_ids",
    }
)
TEACHER_FIELDS = frozenset(
    {"schema_version", "encoding", "token_ids"}
)
MAX_TEACHER_TOKENS = 64
MAX_INPUT_TOKENS = 128


def token_ids_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def load_token_document(
    path: Path,
    *,
    label: str,
    expected_fields: frozenset[str],
) -> tuple[dict[str, Any], list[int]]:
    document = common.load_json(path)
    common.require(isinstance(document, dict), f"{label} must be a JSON object")
    common.require(
        set(document) == expected_fields,
        f"{label} fields differ: {sorted(set(document) ^ expected_fields)}",
    )
    common.require(document["schema_version"] == 1, f"{label} schema_version must be 1")
    common.require(document["encoding"] == "u32-le", f"{label} encoding must be u32-le")
    values = document["token_ids"]
    common.require(isinstance(values, list) and values, f"{label} token_ids must be non-empty")
    tokens: list[int] = []
    for index, value in enumerate(values):
        common.require(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value < model_reference.VOCABULARY_SIZE,
            f"{label} token_ids[{index}] is outside the Qwen3.5 vocabulary",
        )
        tokens.append(value)
    if "token_count" in document:
        common.require(
            document["token_count"] == len(tokens),
            f"{label} token_count differs from token_ids",
        )
    if "token_ids_sha256" in document:
        common.require(
            document["token_ids_sha256"] == token_ids_sha256(tokens),
            f"{label} token_ids_sha256 differs",
        )
    return document, tokens


def validate_reviewed_model(reader: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    common.require(
        len(tensors) == len(reader.tensors) == 426,
        "GGUF tensor inventory differs from the reviewed model",
    )
    metadata: dict[str, Any] = {}
    for key, expected in common.EXPECTED_METADATA.items():
        field = reader.get_field(key)
        common.require(field is not None, f"GGUF is missing metadata: {key}")
        actual = common.json_value(field.contents())
        if isinstance(expected, float):
            common.require(
                math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12),
                f"GGUF metadata {key} differs",
            )
        else:
            common.require(actual == expected, f"GGUF metadata {key} differs")
        metadata[key] = actual
    common.require(
        common.json_value(reader.get_field("qwen35.feed_forward_length").contents())
        == model_reference.INTERMEDIATE_SIZE,
        "GGUF feed-forward size differs",
    )
    return tensors, metadata


def evaluate_final_hidden(
    np: Any,
    tensors: dict[str, Any],
    dequantize: Any,
    token_ids: list[int],
) -> tuple[Any, Any]:
    def weight(name: str) -> Any:
        tensor = tensors[name]
        value = dequantize(tensor.data, tensor.tensor_type).astype(
            np.float32, copy=False
        )
        common.require(
            bool(np.isfinite(value).all()),
            f"dequantized tensor is non-finite: {name}",
        )
        return value

    embedding_tensor = tensors["token_embd.weight"]
    token_index = np.asarray(token_ids, dtype=np.int64)
    hidden = dequantize(
        embedding_tensor.data[token_index].copy(), embedding_tensor.tensor_type
    ).astype(np.float32, copy=False)
    common.require(
        hidden.shape == (len(token_ids), model_reference.HIDDEN_SIZE),
        "reference embedding shape differs",
    )
    for layer in range(model_reference.LAYER_COUNT):
        if (layer + 1) % model_reference.FULL_ATTENTION_INTERVAL == 0:
            attention_residual = model_reference.full_attention_residual(
                np, hidden, layer, weight
            )
        else:
            attention_residual = model_reference.linear_attention_residual(
                np, hidden, layer, weight
            )
        next_hidden = model_reference.finish_dense_layer(
            np, attention_residual, layer, weight
        )
        del attention_residual, hidden
        hidden = next_hidden
        print(
            f"Qwen3.5 teacher FP32 oracle layer "
            f"{layer + 1}/{model_reference.LAYER_COUNT} complete",
            file=sys.stderr,
            flush=True,
        )
        gc.collect()
    final_hidden = model_reference.rms_norm(
        np, hidden, weight("output_norm.weight")
    )
    common.require(
        final_hidden.shape
        == (len(token_ids), model_reference.HIDDEN_SIZE)
        and bool(np.isfinite(final_hidden).all()),
        "reference final hidden state is invalid",
    )
    return embedding_tensor, final_hidden


def stable_top2(np: Any, values: Any) -> tuple[list[int], list[float]]:
    common.require(
        values.ndim == 1 and values.size >= 2 and bool(np.isfinite(values).all()),
        "top-2 input must be a finite vector",
    )
    first = int(np.argmax(values))
    second_values = values.copy()
    second_values[first] = -np.inf
    second = int(np.argmax(second_values))
    return [first, second], [float(values[first]), float(values[second])]


def write_decision_logits(
    np: Any,
    embedding_tensor: Any,
    dequantize: Any,
    decision_hidden: Any,
    output_path: Path,
) -> list[dict[str, Any]]:
    shape = (decision_hidden.shape[0], model_reference.VOCABULARY_SIZE)
    output = np.memmap(output_path, dtype="<f4", mode="w+", shape=shape)
    for start in range(0, model_reference.VOCABULARY_SIZE, model_reference.LOGIT_CHUNK_ROWS):
        end = min(
            model_reference.VOCABULARY_SIZE,
            start + model_reference.LOGIT_CHUNK_ROWS,
        )
        rows = dequantize(
            embedding_tensor.data[start:end].copy(), embedding_tensor.tensor_type
        ).astype(np.float32, copy=False)
        chunk = decision_hidden @ rows.T
        common.require(
            chunk.shape == (shape[0], end - start)
            and bool(np.isfinite(chunk).all()),
            "teacher logits chunk is invalid",
        )
        output[:, start:end] = chunk
        del rows, chunk
    output.flush()
    decisions: list[dict[str, Any]] = []
    for token_index in range(shape[0]):
        top_ids, top_logits = stable_top2(np, output[token_index])
        decisions.append(
            {
                "token_index": token_index,
                "top2_token_ids": top_ids,
                "top2_logits": top_logits,
                "margin": top_logits[0] - top_logits[1],
            }
        )
    del output
    return decisions


def build_reference(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    ferrum_source = common.git_provenance(repo_root, allow_untracked=True)
    llama_root = Path(args.llama_cpp_root).expanduser().resolve()
    common.require(
        (llama_root / "gguf-py/gguf").is_dir(),
        "--llama-cpp-root does not contain gguf-py/gguf",
    )
    llama_source = common.git_provenance(llama_root, allow_untracked=False)
    sys.path.insert(0, str(llama_root / "gguf-py"))
    try:
        import gguf  # type: ignore
        import numpy as np  # type: ignore
        from gguf import GGUFReader  # type: ignore
        from gguf.quants import dequantize  # type: ignore
    except ImportError as error:
        raise common.ReferenceError(
            "numpy, pyyaml, and llama.cpp gguf-py are required; run with "
            "`uv run --with numpy --with pyyaml`"
        ) from error

    model_input = Path(args.model).expanduser().absolute()
    model_path = model_input.resolve()
    prompt_path = Path(args.prompt_token_ids).expanduser().resolve()
    teacher_path = Path(args.teacher_token_ids).expanduser().resolve()
    common.require(
        model_path.is_file() and not model_path.is_symlink(),
        "--model must resolve to a regular GGUF file",
    )
    common.require(
        prompt_path.is_file() and not prompt_path.is_symlink(),
        "--prompt-token-ids must be a regular JSON file",
    )
    common.require(
        teacher_path.is_file() and not teacher_path.is_symlink(),
        "--teacher-token-ids must be a regular JSON file",
    )
    prompt_document, prompt_tokens = load_token_document(
        prompt_path,
        label="prompt token document",
        expected_fields=PROMPT_FIELDS,
    )
    _teacher_document, teacher_tokens = load_token_document(
        teacher_path,
        label="teacher token document",
        expected_fields=TEACHER_FIELDS,
    )
    common.require(
        1 <= len(teacher_tokens) <= MAX_TEACHER_TOKENS,
        f"teacher token count must be within 1..={MAX_TEACHER_TOKENS}",
    )
    full_input_tokens = prompt_tokens + teacher_tokens[:-1]
    common.require(
        len(full_input_tokens) <= MAX_INPUT_TOKENS,
        f"teacher fixture exceeds {MAX_INPUT_TOKENS} input tokens",
    )

    reader = GGUFReader(str(model_path), "r")
    tensors, metadata = validate_reviewed_model(reader)
    embedding_tensor, final_hidden = evaluate_final_hidden(
        np, tensors, dequantize, full_input_tokens
    )
    first_position = len(prompt_tokens) - 1
    decision_positions = list(
        range(first_position, first_position + len(teacher_tokens))
    )
    common.require(
        decision_positions[-1] == len(full_input_tokens) - 1,
        "teacher decision positions do not cover the canonical history",
    )
    decision_hidden = np.ascontiguousarray(
        final_hidden[np.asarray(decision_positions, dtype=np.int64)],
        dtype=np.float32,
    )
    logits_path = out_dir / "decision-logits.f32"
    decisions = write_decision_logits(
        np, embedding_tensor, dequantize, decision_hidden, logits_path
    )
    for token_index, decision in enumerate(decisions):
        decision["position"] = decision_positions[token_index]
        decision["canonical_token_id"] = teacher_tokens[token_index]

    source_path = Path(__file__).resolve()
    model_source_path = Path(model_reference.__file__).resolve()
    linear_source_path = Path(common.__file__).resolve()
    report = {
        "schema_version": 1,
        "status": "pass",
        "oracle": {
            "identity": "cpu.fp32.python.qwen35_gguf_teacher_logits_reference",
            "precision": "fp32",
            "semantics": (
                "independent same-history Qwen3.5 dense-hybrid transformer stack "
                "and tied full-vocabulary head over GGUF-dequantized weights"
            ),
            "source_path": str(source_path.relative_to(repo_root)),
            "source_sha256": common.sha256_file(source_path),
            "model_reference_source_path": str(model_source_path.relative_to(repo_root)),
            "model_reference_source_sha256": common.sha256_file(model_source_path),
            "linear_common_source_path": str(linear_source_path.relative_to(repo_root)),
            "linear_common_source_sha256": common.sha256_file(linear_source_path),
            "ferrum_source": ferrum_source,
            "llama_cpp_gguf_py_source": llama_source,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "gguf_module_path": str(Path(gguf.__file__).resolve()),
        },
        "model": {
            "path": str(model_path),
            "sha256": common.sha256_file(model_path),
            "byte_count": model_path.stat().st_size,
            "format": "GGUF Q4_K_M",
            "hugging_face_snapshot": common.hf_snapshot_identity(model_input),
            "tensor_count": len(tensors),
            "upstream_model": metadata.get("general.base_model.0.repo_url"),
            "quantized_by": metadata.get("general.quantized_by"),
            "license": metadata.get("general.license"),
        },
        "input": {
            "request_id": prompt_document["request_id"],
            "prompt_file": str(prompt_path),
            "prompt_file_sha256": common.sha256_file(prompt_path),
            "prompt_token_count": len(prompt_tokens),
            "prompt_token_ids_sha256": token_ids_sha256(prompt_tokens),
            "teacher_file": str(teacher_path),
            "teacher_file_sha256": common.sha256_file(teacher_path),
            "teacher_token_count": len(teacher_tokens),
            "teacher_token_ids_sha256": token_ids_sha256(teacher_tokens),
            "full_input_token_count": len(full_input_tokens),
            "full_input_token_ids_sha256": token_ids_sha256(full_input_tokens),
            "decision_positions": decision_positions,
        },
        "output": {
            "raw_file": logits_path.name,
            "raw_sha256": common.sha256_file(logits_path),
            "raw_bytes": logits_path.stat().st_size,
            "logical_dtype": "fp32",
            "logical_shape": [len(teacher_tokens), model_reference.VOCABULARY_SIZE],
            "nan_count": 0,
            "inf_count": 0,
            "decisions": decisions,
        },
        "invocation": {"argv": [str(value) for value in sys.argv], "cwd": os.getcwd()},
    }
    common.write_json(out_dir / "report.json", report)
    return report


def self_test() -> None:
    try:
        import numpy as np  # type: ignore
    except ImportError as error:
        raise common.ReferenceError("numpy is required for the self-test") from error
    with tempfile.TemporaryDirectory(prefix="qwen35-teacher-logits-selftest-") as temporary:
        root = Path(temporary)
        prompt_tokens = [11, 12]
        prompt = {
            "schema_version": 1,
            "encoding": "u32-le",
            "request_id": "request.selftest",
            "token_count": len(prompt_tokens),
            "token_ids_sha256": token_ids_sha256(prompt_tokens),
            "token_ids": prompt_tokens,
        }
        prompt_path = root / "prompt.json"
        common.write_json(prompt_path, prompt)
        _document, loaded = load_token_document(
            prompt_path,
            label="prompt token document",
            expected_fields=PROMPT_FIELDS,
        )
        common.require(loaded == prompt_tokens, "prompt token loader differs")
        top_ids, top_logits = stable_top2(
            np, np.asarray([1.0, 3.0, 3.0, -1.0], dtype=np.float32)
        )
        common.require(
            top_ids == [1, 2] and top_logits == [3.0, 3.0],
            "stable top-2 ordering differs",
        )
        prompt["token_ids_sha256"] = "0" * 64
        common.write_json(prompt_path, prompt)
        try:
            load_token_document(
                prompt_path,
                label="prompt token document",
                expected_fields=PROMPT_FIELDS,
            )
        except common.ReferenceError as error:
            common.require(
                "token_ids_sha256 differs" in str(error),
                "tampered prompt rejected for the wrong reason",
            )
        else:
            raise common.ReferenceError("tampered prompt unexpectedly passed")
    print(SELF_TEST_PASS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--prompt-token-ids")
    parser.add_argument("--teacher-token-ids")
    parser.add_argument("--llama-cpp-root")
    parser.add_argument("--out")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        for flag, value in (
            ("--model", args.model),
            ("--prompt-token-ids", args.prompt_token_ids),
            ("--teacher-token-ids", args.teacher_token_ids),
            ("--llama-cpp-root", args.llama_cpp_root),
            ("--out", args.out),
        ):
            common.require(value is not None, f"{flag} is required")
        out_dir = Path(args.out).expanduser().resolve()
        repo_root = Path(__file__).resolve().parents[2]
        common.require(
            not out_dir.is_relative_to(repo_root),
            "reference artifacts must be outside the source tree",
        )
        common.require(
            not out_dir.exists() or not any(out_dir.iterdir()),
            "output directory must be absent or empty",
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        build_reference(args, out_dir)
        print(f"{PASS_PREFIX}: {out_dir}")
        return 0
    except (common.ReferenceError, OSError, ValueError) as error:
        print(f"QWEN35 GGUF TEACHER LOGITS REFERENCE FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
