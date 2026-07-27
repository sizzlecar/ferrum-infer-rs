#!/usr/bin/env python3
"""Validate raw terminal-fence checkpoint artifacts emitted by vNext."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any


PASS_PREFIX = "RUNTIME VNEXT CHECKPOINT ARTIFACT PASS"
COMPARISON_PASS_PREFIX = "RUNTIME VNEXT CHECKPOINT COMPARISON"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
LAYER_VALUE_RE = re.compile(
    r"value\.layer\.(?P<layer>[0-9]+)\."
    r"(?P<stage>attention|post_attention_norm|mlp|output)"
)
ELEMENT_FORMATS = {
    "f16": ("<e", 2),
    "bf16": (None, 2),
    "f32": ("<f", 4),
    "u32": ("<I", 4),
}

PLAN_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "plan_id",
        "plan_hash",
        "model_id",
        "family_fingerprint",
        "program_fingerprint",
        "run_id",
        "maximum_prefill_waves",
        "checkpoints",
    }
)
PLAN_FIELDS_V2 = PLAN_FIELDS_V1 | {"maximum_decode_waves"}
PLAN_FIELDS_V3 = PLAN_FIELDS_V2 | {"capture_product_output"}
WAVE_FIELDS = frozenset(
    {
        "schema_version",
        "capture_index",
        "plan_id",
        "plan_hash",
        "model_id",
        "family_fingerprint",
        "program_fingerprint",
        "run_id",
        "wave_kind",
        "participant_count",
        "completion_fingerprint",
        "receipt_fingerprint",
        "records",
    }
)
WAVE_FIELDS_V3 = WAVE_FIELDS | {"product_outputs"}
CHECKPOINT_FIELDS = frozenset(
    {
        "value_id",
        "producer_node_id",
        "output_ordinal",
        "resource_id",
        "logical_offset_bytes",
        "tensor",
    }
)
TENSOR_FIELDS = frozenset({"dimensions", "element_type", "layout"})
RECORD_FIELDS = frozenset(
    {
        "value",
        "participant_index",
        "request_id",
        "token_span",
        "output_layout",
        "raw_file",
        "raw_bytes",
        "raw_sha256",
    }
)
PRODUCT_OUTPUT_FIELDS = frozenset(
    {
        "output_mode",
        "node_id",
        "resource_id",
        "logical_offset_bytes",
        "participant_index",
        "request_id",
        "token_span",
        "output_layout",
        "raw_file",
        "raw_bytes",
        "raw_sha256",
    }
)
TOKEN_SPAN_FIELDS = frozenset(
    {
        "immediate_tokens",
        "full_input_tokens",
        "fit_input_tokens",
        "immediate_start_token",
        "immediate_end_token",
        "fingerprint",
    }
)
OUTPUT_LAYOUT_FIELDS = frozenset({"element_type", "element_count"})
IDENTITY_FIELDS = (
    "plan_id",
    "plan_hash",
    "model_id",
    "family_fingerprint",
    "program_fingerprint",
    "run_id",
)


class ArtifactError(RuntimeError):
    """The checkpoint artifact is malformed or incomplete."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ArtifactError(f"non-finite JSON constant is forbidden: {value}")


def load_json(path: Path) -> Any:
    try:
        payload = path.read_text(encoding="utf-8")
        return json.loads(
            payload,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except ArtifactError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"cannot load {path}: {error}") from error


def exact_object(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    require(not fields - actual, f"{label} is missing fields: {sorted(fields - actual)}")
    require(not actual - fields, f"{label} has unknown fields: {sorted(actual - fields)}")
    return value


def text(value: Any, label: str) -> str:
    require(isinstance(value, str) and value == value.strip() and bool(value),
            f"{label} must be a non-empty trimmed string")
    return value


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    require(value >= minimum, f"{label} must be >= {minimum}")
    return value


def boolean(value: Any, label: str) -> bool:
    require(isinstance(value, bool), f"{label} must be a boolean")
    return value


def sha256(value: Any, label: str) -> str:
    digest = text(value, label)
    require(SHA256_RE.fullmatch(digest) is not None, f"{label} must be a lowercase SHA256")
    return digest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_tensor(value: Any, label: str) -> tuple[str, int]:
    tensor = exact_object(value, TENSOR_FIELDS, label)
    dimensions = tensor["dimensions"]
    require(isinstance(dimensions, list) and dimensions, f"{label}.dimensions must be non-empty")
    element_capacity = 1
    for index, dimension in enumerate(dimensions):
        element_capacity *= integer(dimension, f"{label}.dimensions[{index}]", minimum=1)
    element_type = text(tensor["element_type"], f"{label}.element_type")
    require(element_type in ELEMENT_FORMATS, f"{label}.element_type is unsupported")
    require(tensor["layout"] == "contiguous", f"{label}.layout must be contiguous")
    return element_type, element_capacity


def validate_checkpoint(value: Any, label: str) -> dict[str, Any]:
    checkpoint = exact_object(value, CHECKPOINT_FIELDS, label)
    text(checkpoint["value_id"], f"{label}.value_id")
    text(checkpoint["producer_node_id"], f"{label}.producer_node_id")
    integer(checkpoint["output_ordinal"], f"{label}.output_ordinal")
    text(checkpoint["resource_id"], f"{label}.resource_id")
    integer(checkpoint["logical_offset_bytes"], f"{label}.logical_offset_bytes")
    validate_tensor(checkpoint["tensor"], f"{label}.tensor")
    return checkpoint


def tensor_stats(path: Path, element_type: str, element_count: int) -> dict[str, Any]:
    fmt, width = ELEMENT_FORMATS[element_type]
    payload = path.read_bytes()
    require(len(payload) == element_count * width, f"{path.name} byte count differs from layout")
    if element_type == "bf16":
        values = (
            struct.unpack("<f", struct.pack("<I", item[0] << 16))[0]
            for item in struct.iter_unpack("<H", payload)
        )
    else:
        assert fmt is not None
        values = (item[0] for item in struct.iter_unpack(fmt, payload))
    if element_type == "u32":
        nonzero = sum(value != 0 for value in values)
        return {"element_count": element_count, "nonzero_count": nonzero}

    minimum = math.inf
    maximum = -math.inf
    finite_count = 0
    nan_count = 0
    inf_count = 0
    nonzero_count = 0
    sum_squares = 0.0
    for value in values:
        if math.isnan(value):
            nan_count += 1
        elif math.isinf(value):
            inf_count += 1
        else:
            finite_count += 1
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            nonzero_count += value != 0.0
            sum_squares += float(value) * float(value)
    require(nan_count == 0 and inf_count == 0, f"{path.name} contains NaN or Inf")
    require(finite_count == element_count, f"{path.name} finite element count differs from layout")
    require(nonzero_count > 0, f"{path.name} is entirely zero")
    return {
        "element_count": element_count,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nonzero_count": nonzero_count,
        "minimum": minimum,
        "maximum": maximum,
        "l2_norm": math.sqrt(sum_squares),
    }


def validate_artifact(
    capture_dir: Path,
    expected_model_id: str | None,
    expected_values: list[str],
) -> dict[str, Any]:
    require(capture_dir.is_dir() and not capture_dir.is_symlink(),
            "capture directory must be a real directory")
    plan_path = capture_dir / "plan.json"
    require(plan_path.is_file() and not plan_path.is_symlink(), "plan.json must be a real file")
    raw_plan = load_json(plan_path)
    require(isinstance(raw_plan, dict), "plan must be an object")
    schema_version = integer(raw_plan.get("schema_version"), "plan.schema_version", minimum=1)
    require(schema_version in (1, 2, 3), "plan.schema_version must be 1, 2, or 3")
    plan_fields = {
        1: PLAN_FIELDS_V1,
        2: PLAN_FIELDS_V2,
        3: PLAN_FIELDS_V3,
    }[schema_version]
    plan = exact_object(
        raw_plan,
        plan_fields,
        "plan",
    )
    plan_hash = sha256(plan["plan_hash"], "plan.plan_hash")
    require(
        plan["plan_id"] == f"plan/sha256/{plan_hash}",
        "plan.plan_id must derive from plan_hash",
    )
    sha256(plan["family_fingerprint"], "plan.family_fingerprint")
    sha256(plan["program_fingerprint"], "plan.program_fingerprint")
    text(plan["run_id"], "plan.run_id")
    model_id = text(plan["model_id"], "plan.model_id")
    if expected_model_id is not None:
        require(model_id == expected_model_id, "plan.model_id differs from --expected-model-id")
    maximum_waves = integer(plan["maximum_prefill_waves"], "plan.maximum_prefill_waves", minimum=1)
    require(maximum_waves <= 16, "plan.maximum_prefill_waves exceeds the product limit")
    maximum_decode_waves = (
        integer(plan["maximum_decode_waves"], "plan.maximum_decode_waves")
        if schema_version >= 2
        else 0
    )
    require(
        maximum_decode_waves <= 512,
        "plan.maximum_decode_waves exceeds the product limit",
    )
    capture_product_output = (
        boolean(plan["capture_product_output"], "plan.capture_product_output")
        if schema_version >= 3
        else False
    )
    checkpoints_raw = plan["checkpoints"]
    require(
        isinstance(checkpoints_raw, list)
        and (bool(checkpoints_raw) or capture_product_output),
        "plan must capture product output or contain checkpoints",
    )
    checkpoints = [
        validate_checkpoint(item, f"plan.checkpoints[{index}]")
        for index, item in enumerate(checkpoints_raw)
    ]
    require(len(checkpoints) <= 63, "plan.checkpoints exceeds the product limit")
    value_ids = [item["value_id"] for item in checkpoints]
    require(value_ids == sorted(set(value_ids)), "plan.checkpoints must be unique and sorted")
    if expected_values:
        require(value_ids == sorted(set(expected_values)),
                "plan checkpoint values differ from --expected-value")
    checkpoint_by_id = {item["value_id"]: item for item in checkpoints}

    prefill_wave_paths = sorted(capture_dir.glob("wave-*.json"))
    decode_wave_paths = sorted(capture_dir.glob("decode-wave-*.json"))
    require(
        len(prefill_wave_paths) == maximum_waves,
        "captured prefill wave count differs from plan maximum",
    )
    require(
        len(decode_wave_paths) == maximum_decode_waves,
        "captured decode wave count differs from plan maximum",
    )
    wave_groups = (
        ("prefill", "wave-", prefill_wave_paths),
        ("decode", "decode-wave-", decode_wave_paths),
    )
    wave_entries = [
        (kind, filename_prefix, index, path)
        for kind, filename_prefix, paths in wave_groups
        for index, path in enumerate(paths)
    ]
    summaries: list[dict[str, Any]] = []
    referenced_raw: set[str] = set()
    for expected_kind, filename_prefix, expected_index, wave_path in wave_entries:
        require(
            wave_path.is_file() and not wave_path.is_symlink(),
            "wave manifest must be a real file",
        )
        wave = exact_object(
            load_json(wave_path),
            WAVE_FIELDS_V3 if schema_version >= 3 else WAVE_FIELDS,
            f"wave[{expected_index}]",
        )
        require(
            wave["schema_version"] == schema_version,
            "wave.schema_version must match plan.schema_version",
        )
        require(wave["capture_index"] == expected_index, "wave.capture_index is not contiguous")
        require(
            wave_path.name == f"{filename_prefix}{expected_index:04}.json",
            "wave filename is not canonical",
        )
        for field in IDENTITY_FIELDS:
            require(wave[field] == plan[field], f"wave.{field} differs from plan")
        require(
            wave["wave_kind"] == expected_kind,
            f"wave.wave_kind must be {expected_kind}",
        )
        participant_count = integer(wave["participant_count"], "wave.participant_count", minimum=1)
        sha256(wave["completion_fingerprint"], "wave.completion_fingerprint")
        sha256(wave["receipt_fingerprint"], "wave.receipt_fingerprint")
        records_raw = wave["records"]
        require(isinstance(records_raw, list), "wave.records must be a list")
        require(len(records_raw) == len(checkpoints) * participant_count,
                "wave record count differs from checkpoint x participant count")
        record_keys: list[tuple[str, int]] = []
        participant_identity: dict[int, tuple[str, Any]] = {}
        records_summary: list[dict[str, Any]] = []
        for record_index, record_raw in enumerate(records_raw):
            label = f"wave.records[{record_index}]"
            record = exact_object(record_raw, RECORD_FIELDS, label)
            value = validate_checkpoint(record["value"], f"{label}.value")
            value_id = value["value_id"]
            require(value_id in checkpoint_by_id and value == checkpoint_by_id[value_id],
                    f"{label}.value differs from plan checkpoint")
            participant_index = integer(record["participant_index"], f"{label}.participant_index")
            require(
                participant_index < participant_count,
                f"{label}.participant_index is out of range",
            )
            text(record["request_id"], f"{label}.request_id")
            token_span = exact_object(
                record["token_span"], TOKEN_SPAN_FIELDS, f"{label}.token_span"
            )
            immediate_tokens = integer(
                token_span["immediate_tokens"],
                f"{label}.token_span.immediate_tokens",
                minimum=1,
            )
            full_tokens = integer(
                token_span["full_input_tokens"],
                f"{label}.token_span.full_input_tokens",
                minimum=1,
            )
            fit_tokens = integer(
                token_span["fit_input_tokens"],
                f"{label}.token_span.fit_input_tokens",
                minimum=1,
            )
            start = integer(
                token_span["immediate_start_token"],
                f"{label}.token_span.immediate_start_token",
            )
            end = integer(
                token_span["immediate_end_token"],
                f"{label}.token_span.immediate_end_token",
                minimum=1,
            )
            require(
                end - start == immediate_tokens
                and full_tokens <= fit_tokens
                and end <= full_tokens,
                f"{label}.token_span is inconsistent",
            )
            sha256(token_span["fingerprint"], f"{label}.token_span.fingerprint")
            identity = (record["request_id"], token_span)
            previous_identity = participant_identity.setdefault(participant_index, identity)
            require(
                previous_identity == identity,
                f"participant {participant_index} request or token span differs across checkpoints",
            )
            layout = exact_object(
                record["output_layout"],
                OUTPUT_LAYOUT_FIELDS,
                f"{label}.output_layout",
            )
            element_type = text(layout["element_type"], f"{label}.output_layout.element_type")
            element_count = integer(
                layout["element_count"],
                f"{label}.output_layout.element_count",
                minimum=1,
            )
            plan_type, plan_capacity = validate_tensor(value["tensor"], f"{label}.value.tensor")
            require(element_type == plan_type and element_count <= plan_capacity,
                    f"{label}.output_layout exceeds or differs from plan tensor")
            raw_name = text(record["raw_file"], f"{label}.raw_file")
            require(Path(raw_name).name == raw_name, f"{label}.raw_file must be a basename")
            raw_path = capture_dir / raw_name
            require(
                raw_path.is_file() and not raw_path.is_symlink(),
                f"{label}.raw_file must be a real file",
            )
            raw_bytes = integer(record["raw_bytes"], f"{label}.raw_bytes", minimum=1)
            require(raw_path.stat().st_size == raw_bytes, f"{label}.raw_bytes differs from file")
            digest = file_sha256(raw_path)
            require(digest == sha256(record["raw_sha256"], f"{label}.raw_sha256"),
                    f"{label}.raw_sha256 differs from file")
            require(
                raw_name not in referenced_raw,
                f"raw file is referenced more than once: {raw_name}",
            )
            referenced_raw.add(raw_name)
            record_keys.append((value_id, participant_index))
            records_summary.append(
                {
                    "value_id": value_id,
                    "participant_index": participant_index,
                    "request_id": record["request_id"],
                    "token_span_fingerprint": token_span["fingerprint"],
                    "raw_file": raw_name,
                    "raw_bytes": raw_bytes,
                    "raw_sha256": digest,
                    "element_type": element_type,
                    "stats": tensor_stats(raw_path, element_type, element_count),
                }
            )
        require(record_keys == sorted(set(record_keys)), "wave records must be unique and sorted")
        product_outputs_raw = wave["product_outputs"] if schema_version >= 3 else []
        require(
            isinstance(product_outputs_raw, list),
            "wave.product_outputs must be a list",
        )
        expected_product_outputs = participant_count if capture_product_output else 0
        require(
            len(product_outputs_raw) == expected_product_outputs,
            "wave product-output count differs from capture policy x participant count",
        )
        product_output_keys: list[int] = []
        product_outputs_summary: list[dict[str, Any]] = []
        for product_index, product_raw in enumerate(product_outputs_raw):
            label = f"wave.product_outputs[{product_index}]"
            product = exact_object(product_raw, PRODUCT_OUTPUT_FIELDS, label)
            output_mode = text(product["output_mode"], f"{label}.output_mode")
            require(
                output_mode in ("full-logits", "greedy-token"),
                f"{label}.output_mode is unsupported",
            )
            text(product["node_id"], f"{label}.node_id")
            text(product["resource_id"], f"{label}.resource_id")
            logical_offset_bytes = integer(
                product["logical_offset_bytes"],
                f"{label}.logical_offset_bytes",
            )
            participant_index = integer(
                product["participant_index"],
                f"{label}.participant_index",
            )
            require(
                participant_index < participant_count,
                f"{label}.participant_index is out of range",
            )
            text(product["request_id"], f"{label}.request_id")
            token_span = exact_object(
                product["token_span"],
                TOKEN_SPAN_FIELDS,
                f"{label}.token_span",
            )
            immediate_tokens = integer(
                token_span["immediate_tokens"],
                f"{label}.token_span.immediate_tokens",
                minimum=1,
            )
            full_tokens = integer(
                token_span["full_input_tokens"],
                f"{label}.token_span.full_input_tokens",
                minimum=1,
            )
            fit_tokens = integer(
                token_span["fit_input_tokens"],
                f"{label}.token_span.fit_input_tokens",
                minimum=1,
            )
            start = integer(
                token_span["immediate_start_token"],
                f"{label}.token_span.immediate_start_token",
            )
            end = integer(
                token_span["immediate_end_token"],
                f"{label}.token_span.immediate_end_token",
                minimum=1,
            )
            require(
                end - start == immediate_tokens
                and full_tokens <= fit_tokens
                and end <= full_tokens,
                f"{label}.token_span is inconsistent",
            )
            sha256(token_span["fingerprint"], f"{label}.token_span.fingerprint")
            identity = (product["request_id"], token_span)
            previous_identity = participant_identity.setdefault(participant_index, identity)
            require(
                previous_identity == identity,
                f"participant {participant_index} request or token span differs across outputs",
            )
            layout = exact_object(
                product["output_layout"],
                OUTPUT_LAYOUT_FIELDS,
                f"{label}.output_layout",
            )
            element_type = text(
                layout["element_type"],
                f"{label}.output_layout.element_type",
            )
            element_count = integer(
                layout["element_count"],
                f"{label}.output_layout.element_count",
                minimum=1,
            )
            if output_mode == "full-logits":
                require(
                    element_type in ("f16", "bf16", "f32"),
                    f"{label} full logits must use a floating element type",
                )
            else:
                require(
                    element_type == "u32" and element_count == 1,
                    f"{label} greedy token must use u32[1]",
                )
            raw_name = text(product["raw_file"], f"{label}.raw_file")
            require(
                Path(raw_name).name == raw_name,
                f"{label}.raw_file must be a basename",
            )
            raw_path = capture_dir / raw_name
            require(
                raw_path.is_file() and not raw_path.is_symlink(),
                f"{label}.raw_file must be a real file",
            )
            raw_bytes = integer(
                product["raw_bytes"],
                f"{label}.raw_bytes",
                minimum=1,
            )
            require(
                raw_path.stat().st_size == raw_bytes,
                f"{label}.raw_bytes differs from file",
            )
            digest = file_sha256(raw_path)
            require(
                digest
                == sha256(product["raw_sha256"], f"{label}.raw_sha256"),
                f"{label}.raw_sha256 differs from file",
            )
            require(
                raw_name not in referenced_raw,
                f"raw file is referenced more than once: {raw_name}",
            )
            referenced_raw.add(raw_name)
            product_output_keys.append(participant_index)
            product_outputs_summary.append(
                {
                    "output_mode": output_mode,
                    "node_id": product["node_id"],
                    "resource_id": product["resource_id"],
                    "logical_offset_bytes": logical_offset_bytes,
                    "participant_index": participant_index,
                    "request_id": product["request_id"],
                    "token_span_fingerprint": token_span["fingerprint"],
                    "raw_file": raw_name,
                    "raw_bytes": raw_bytes,
                    "raw_sha256": digest,
                    "element_type": element_type,
                    "stats": tensor_stats(raw_path, element_type, element_count),
                }
            )
        require(
            product_output_keys == list(range(expected_product_outputs)),
            "wave product outputs must be unique and sorted by participant",
        )
        summaries.append(
            {
                "wave_kind": expected_kind,
                "capture_index": expected_index,
                "participant_count": participant_count,
                "records": records_summary,
                "product_outputs": product_outputs_summary,
            }
        )

    actual_raw = {path.name for path in capture_dir.glob("*.bin")}
    require(actual_raw == referenced_raw, "raw file set differs from manifest references")
    return {
        "schema_version": schema_version,
        "status": "pass",
        "capture_dir": str(capture_dir.resolve()),
        "model_id": model_id,
        "plan_id": plan["plan_id"],
        "plan_hash": plan_hash,
        "family_fingerprint": plan["family_fingerprint"],
        "program_fingerprint": plan["program_fingerprint"],
        "run_id": plan["run_id"],
        "checkpoint_values": value_ids,
        "capture_product_output": capture_product_output,
        "maximum_prefill_waves": maximum_waves,
        "maximum_decode_waves": maximum_decode_waves,
        "wave_count": len(summaries),
        "prefill_wave_count": len(prefill_wave_paths),
        "decode_wave_count": len(decode_wave_paths),
        "waves": summaries,
    }


def semantic_value_order(value_id: str) -> tuple[int, int, int, str]:
    layer_match = LAYER_VALUE_RE.fullmatch(value_id)
    if layer_match is not None:
        stage_order = {
            "attention": 0,
            "post_attention_norm": 1,
            "mlp": 2,
            "output": 3,
        }
        return (
            0,
            int(layer_match.group("layer")),
            stage_order[layer_match.group("stage")],
            value_id,
        )
    output_order = {
        "value.output.final_hidden": 0,
        "value.output.logits": 1,
        "value.output.greedy_token": 2,
    }
    return (1, output_order.get(value_id, 3), 0, value_id)


def compare_artifacts(
    baseline_dir: Path,
    candidate_dir: Path,
    expected_model_id: str | None,
    expected_values: list[str],
    expected_relation: str,
) -> dict[str, Any]:
    baseline = validate_artifact(baseline_dir, expected_model_id, expected_values)
    candidate = validate_artifact(candidate_dir, expected_model_id, expected_values)
    for field in (
        "model_id",
        "family_fingerprint",
        "program_fingerprint",
        "checkpoint_values",
        "capture_product_output",
        "maximum_prefill_waves",
        "maximum_decode_waves",
    ):
        require(
            baseline[field] == candidate[field],
            f"comparison {field} differs across captures",
        )

    baseline_waves = {
        (wave["wave_kind"], wave["capture_index"]): wave for wave in baseline["waves"]
    }
    candidate_waves = {
        (wave["wave_kind"], wave["capture_index"]): wave for wave in candidate["waves"]
    }
    require(
        baseline_waves.keys() == candidate_waves.keys(),
        "comparison wave topology differs across captures",
    )

    first_difference: dict[str, Any] | None = None
    wave_order = {"prefill": 0, "decode": 1}
    for wave_key in sorted(
        baseline_waves,
        key=lambda item: (wave_order[item[0]], item[1]),
    ):
        baseline_wave = baseline_waves[wave_key]
        candidate_wave = candidate_waves[wave_key]
        require(
            baseline_wave["participant_count"] == candidate_wave["participant_count"],
            f"comparison participant count differs at {wave_key}",
        )
        baseline_records = {
            (record["value_id"], record["participant_index"]): record
            for record in baseline_wave["records"]
        }
        candidate_records = {
            (record["value_id"], record["participant_index"]): record
            for record in candidate_wave["records"]
        }
        require(
            baseline_records.keys() == candidate_records.keys(),
            f"comparison checkpoint topology differs at {wave_key}",
        )
        for record_key in sorted(
            baseline_records,
            key=lambda item: (semantic_value_order(item[0]), item[1]),
        ):
            baseline_record = baseline_records[record_key]
            candidate_record = candidate_records[record_key]
            for field in ("raw_bytes", "element_type"):
                require(
                    baseline_record[field] == candidate_record[field],
                    f"comparison {field} differs at {wave_key}/{record_key}",
                )
            if (
                baseline_record["token_span_fingerprint"]
                != candidate_record["token_span_fingerprint"]
            ):
                first_difference = {
                    "difference_kind": "input-token-span",
                    "wave_kind": wave_key[0],
                    "capture_index": wave_key[1],
                    "value_id": record_key[0],
                    "participant_index": record_key[1],
                    "baseline_token_span_fingerprint": baseline_record[
                        "token_span_fingerprint"
                    ],
                    "candidate_token_span_fingerprint": candidate_record[
                        "token_span_fingerprint"
                    ],
                }
                break
            if baseline_record["raw_sha256"] != candidate_record["raw_sha256"]:
                first_difference = {
                    "difference_kind": "semantic-value",
                    "wave_kind": wave_key[0],
                    "capture_index": wave_key[1],
                    "value_id": record_key[0],
                    "participant_index": record_key[1],
                    "baseline_raw_sha256": baseline_record["raw_sha256"],
                    "candidate_raw_sha256": candidate_record["raw_sha256"],
                    "baseline_raw_file": baseline_record["raw_file"],
                    "candidate_raw_file": candidate_record["raw_file"],
                }
                break
        if first_difference is not None:
            break
        baseline_products = {
            record["participant_index"]: record
            for record in baseline_wave["product_outputs"]
        }
        candidate_products = {
            record["participant_index"]: record
            for record in candidate_wave["product_outputs"]
        }
        require(
            baseline_products.keys() == candidate_products.keys(),
            f"comparison product-output topology differs at {wave_key}",
        )
        for participant_index in sorted(baseline_products):
            baseline_product = baseline_products[participant_index]
            candidate_product = candidate_products[participant_index]
            for field in ("output_mode", "raw_bytes", "element_type"):
                require(
                    baseline_product[field] == candidate_product[field],
                    f"comparison product {field} differs at {wave_key}/{participant_index}",
                )
            if (
                baseline_product["token_span_fingerprint"]
                != candidate_product["token_span_fingerprint"]
            ):
                first_difference = {
                    "difference_kind": "input-token-span",
                    "wave_kind": wave_key[0],
                    "capture_index": wave_key[1],
                    "output_mode": baseline_product["output_mode"],
                    "participant_index": participant_index,
                    "baseline_token_span_fingerprint": baseline_product[
                        "token_span_fingerprint"
                    ],
                    "candidate_token_span_fingerprint": candidate_product[
                        "token_span_fingerprint"
                    ],
                }
                break
            if baseline_product["raw_sha256"] != candidate_product["raw_sha256"]:
                first_difference = {
                    "difference_kind": "product-output",
                    "wave_kind": wave_key[0],
                    "capture_index": wave_key[1],
                    "output_mode": baseline_product["output_mode"],
                    "participant_index": participant_index,
                    "baseline_raw_sha256": baseline_product["raw_sha256"],
                    "candidate_raw_sha256": candidate_product["raw_sha256"],
                    "baseline_raw_file": baseline_product["raw_file"],
                    "candidate_raw_file": candidate_product["raw_file"],
                }
                break
        if first_difference is not None:
            break

    actual_relation = "different" if first_difference is not None else "equal"
    require(
        actual_relation == expected_relation,
        f"checkpoint relation is {actual_relation}, expected {expected_relation}; "
        f"first_difference={first_difference}",
    )
    return {
        "schema_version": 1,
        "status": "pass",
        "expected_relation": expected_relation,
        "actual_relation": actual_relation,
        "baseline_capture_dir": baseline["capture_dir"],
        "candidate_capture_dir": candidate["capture_dir"],
        "model_id": baseline["model_id"],
        "baseline_plan_id": baseline["plan_id"],
        "candidate_plan_id": candidate["plan_id"],
        "program_fingerprint": baseline["program_fingerprint"],
        "baseline_run_id": baseline["run_id"],
        "candidate_run_id": candidate["run_id"],
        "first_difference": first_difference,
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        capture = Path(temporary) / "capture"
        capture.mkdir()
        raw = struct.pack("<ee", 1.0, -0.5)
        raw_name = "capture-0000-participant-0000-value_test-000000000000.bin"
        (capture / raw_name).write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        decode_raw = struct.pack("<ee", 0.25, -2.0)
        decode_raw_name = "decode-capture-0000-participant-0000-value_test-000000000000.bin"
        (capture / decode_raw_name).write_bytes(decode_raw)
        decode_digest = hashlib.sha256(decode_raw).hexdigest()
        checkpoint = {
            "value_id": "value.test",
            "producer_node_id": "node.test",
            "output_ordinal": 0,
            "resource_id": "resource/test",
            "logical_offset_bytes": 0,
            "tensor": {"dimensions": [2], "element_type": "f16", "layout": "contiguous"},
        }
        identity = {
            "plan_id": f"plan/sha256/{'1' * 64}",
            "plan_hash": "1" * 64,
            "model_id": "model.test",
            "family_fingerprint": "2" * 64,
            "program_fingerprint": "3" * 64,
            "run_id": "run.test",
        }
        plan = {
            "schema_version": 2,
            **identity,
            "maximum_prefill_waves": 1,
            "maximum_decode_waves": 1,
            "checkpoints": [checkpoint],
        }
        wave = {
            "schema_version": 2,
            "capture_index": 0,
            **identity,
            "wave_kind": "prefill",
            "participant_count": 1,
            "completion_fingerprint": "4" * 64,
            "receipt_fingerprint": "5" * 64,
            "records": [
                {
                    "value": checkpoint,
                    "participant_index": 0,
                    "request_id": "request.test",
                    "token_span": {
                        "immediate_tokens": 2,
                        "full_input_tokens": 2,
                        "fit_input_tokens": 128,
                        "immediate_start_token": 0,
                        "immediate_end_token": 2,
                        "fingerprint": "6" * 64,
                    },
                    "output_layout": {"element_type": "f16", "element_count": 2},
                    "raw_file": raw_name,
                    "raw_bytes": len(raw),
                    "raw_sha256": digest,
                }
            ],
        }
        decode_wave = {
            **wave,
            "wave_kind": "decode",
            "records": [
                {
                    **wave["records"][0],
                    "raw_file": decode_raw_name,
                    "raw_bytes": len(decode_raw),
                    "raw_sha256": decode_digest,
                }
            ],
        }
        (capture / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        (capture / "wave-0000.json").write_text(json.dumps(wave), encoding="utf-8")
        (capture / "decode-wave-0000.json").write_text(
            json.dumps(decode_wave), encoding="utf-8"
        )
        validate_artifact(capture, "model.test", ["value.test"])

        candidate = Path(temporary) / "candidate"
        shutil.copytree(capture, candidate)
        candidate_plan = load_json(candidate / "plan.json")
        candidate_plan["plan_hash"] = "7" * 64
        candidate_plan["plan_id"] = f"plan/sha256/{'7' * 64}"
        (candidate / "plan.json").write_text(
            json.dumps(candidate_plan), encoding="utf-8"
        )
        for candidate_wave_path in (
            candidate / "wave-0000.json",
            candidate / "decode-wave-0000.json",
        ):
            candidate_wave = load_json(candidate_wave_path)
            candidate_wave["plan_hash"] = "7" * 64
            candidate_wave["plan_id"] = f"plan/sha256/{'7' * 64}"
            candidate_wave_path.write_text(
                json.dumps(candidate_wave), encoding="utf-8"
            )
        candidate_raw = struct.pack("<ee", 0.5, -2.0)
        (candidate / decode_raw_name).write_bytes(candidate_raw)
        candidate_decode_wave = load_json(candidate / "decode-wave-0000.json")
        candidate_decode_wave["records"][0]["raw_sha256"] = hashlib.sha256(
            candidate_raw
        ).hexdigest()
        (candidate / "decode-wave-0000.json").write_text(
            json.dumps(candidate_decode_wave), encoding="utf-8"
        )
        different = compare_artifacts(
            capture,
            candidate,
            "model.test",
            ["value.test"],
            "different",
        )
        require(
            different["first_difference"]["difference_kind"] == "semantic-value"
            and different["first_difference"]["wave_kind"] == "decode"
            and different["first_difference"]["capture_index"] == 0
            and different["first_difference"]["value_id"] == "value.test",
            "comparison did not identify the first decode semantic-value difference",
        )
        equal = compare_artifacts(
            capture,
            capture,
            "model.test",
            ["value.test"],
            "equal",
        )
        require(
            equal["first_difference"] is None,
            "equal comparison reported a difference",
        )

        (capture / raw_name).write_bytes(raw + b"bad")
        try:
            validate_artifact(capture, "model.test", ["value.test"])
        except ArtifactError as error:
            require("raw_bytes differs" in str(error), "self-test rejected the wrong mutation")
        else:
            raise ArtifactError("self-test accepted a mutated raw tensor")

        (capture / raw_name).write_bytes(raw)
        (capture / decode_raw_name).unlink()
        (capture / "decode-wave-0000.json").unlink()
        plan["schema_version"] = 1
        plan.pop("maximum_decode_waves")
        wave["schema_version"] = 1
        (capture / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        (capture / "wave-0000.json").write_text(json.dumps(wave), encoding="utf-8")
        validate_artifact(capture, "model.test", ["value.test"])

        product_capture = Path(temporary) / "product-capture"
        product_capture.mkdir()
        product_raw = struct.pack("<ee", 3.0, -1.0)
        product_name = "product-output-0000-participant-0000-full-logits.bin"
        (product_capture / product_name).write_bytes(product_raw)
        product_digest = hashlib.sha256(product_raw).hexdigest()
        product_decode_raw = struct.pack("<ee", 2.0, -0.5)
        product_decode_name = (
            "decode-product-output-0000-participant-0000-full-logits.bin"
        )
        (product_capture / product_decode_name).write_bytes(product_decode_raw)
        product_decode_digest = hashlib.sha256(product_decode_raw).hexdigest()
        product_identity = {
            **identity,
            "plan_id": f"plan/sha256/{'8' * 64}",
            "plan_hash": "8" * 64,
            "run_id": "run.product",
        }
        product_plan = {
            "schema_version": 3,
            **product_identity,
            "maximum_prefill_waves": 1,
            "maximum_decode_waves": 1,
            "capture_product_output": True,
            "checkpoints": [],
        }
        product_record = {
            "output_mode": "full-logits",
            "node_id": "node.logits",
            "resource_id": "resource/step/0",
            "logical_offset_bytes": 0,
            "participant_index": 0,
            "request_id": "request.product",
            "token_span": {
                "immediate_tokens": 2,
                "full_input_tokens": 2,
                "fit_input_tokens": 128,
                "immediate_start_token": 0,
                "immediate_end_token": 2,
                "fingerprint": "9" * 64,
            },
            "output_layout": {"element_type": "f16", "element_count": 2},
            "raw_file": product_name,
            "raw_bytes": len(product_raw),
            "raw_sha256": product_digest,
        }
        product_wave = {
            "schema_version": 3,
            "capture_index": 0,
            **product_identity,
            "wave_kind": "prefill",
            "participant_count": 1,
            "completion_fingerprint": "a" * 64,
            "receipt_fingerprint": "b" * 64,
            "records": [],
            "product_outputs": [product_record],
        }
        product_decode_wave = {
            **product_wave,
            "wave_kind": "decode",
            "product_outputs": [
                {
                    **product_record,
                    "raw_file": product_decode_name,
                    "raw_bytes": len(product_decode_raw),
                    "raw_sha256": product_decode_digest,
                }
            ],
        }
        (product_capture / "plan.json").write_text(
            json.dumps(product_plan),
            encoding="utf-8",
        )
        (product_capture / "wave-0000.json").write_text(
            json.dumps(product_wave),
            encoding="utf-8",
        )
        (product_capture / "decode-wave-0000.json").write_text(
            json.dumps(product_decode_wave),
            encoding="utf-8",
        )
        validated_product = validate_artifact(product_capture, "model.test", [])
        require(
            validated_product["capture_product_output"]
            and not validated_product["checkpoint_values"],
            "product-only capture did not validate without retained checkpoints",
        )

        product_candidate = Path(temporary) / "product-candidate"
        shutil.copytree(product_capture, product_candidate)
        changed_product = struct.pack("<ee", 2.5, -0.5)
        (product_candidate / product_decode_name).write_bytes(changed_product)
        changed_wave = load_json(product_candidate / "decode-wave-0000.json")
        changed_wave["product_outputs"][0]["raw_sha256"] = hashlib.sha256(
            changed_product
        ).hexdigest()
        (product_candidate / "decode-wave-0000.json").write_text(
            json.dumps(changed_wave),
            encoding="utf-8",
        )
        product_difference = compare_artifacts(
            product_capture,
            product_candidate,
            "model.test",
            [],
            "different",
        )
        require(
            product_difference["first_difference"]["difference_kind"]
            == "product-output"
            and product_difference["first_difference"]["wave_kind"] == "decode"
            and product_difference["first_difference"]["capture_index"] == 0,
            "comparison did not identify the first product-output difference",
        )
    print("RUNTIME VNEXT CHECKPOINT ARTIFACT SELF-TEST PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", nargs="?", type=Path)
    parser.add_argument("--expected-model-id")
    parser.add_argument("--expected-value", action="append", default=[])
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--compare-capture", type=Path)
    parser.add_argument("--expected-relation", choices=("equal", "different"))
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        require(args.capture_dir is not None, "capture_dir is required")
        if args.compare_capture is not None:
            require(
                args.expected_relation is not None,
                "--expected-relation is required with --compare-capture",
            )
            summary = compare_artifacts(
                args.capture_dir,
                args.compare_capture,
                args.expected_model_id,
                args.expected_value,
                args.expected_relation,
            )
        else:
            require(
                args.expected_relation is None,
                "--expected-relation requires --compare-capture",
            )
            summary = validate_artifact(
                args.capture_dir,
                args.expected_model_id,
                args.expected_value,
            )
        if args.summary is not None:
            write_summary(args.summary, summary)
        if args.compare_capture is not None:
            print(
                f"{COMPARISON_PASS_PREFIX} {args.expected_relation.upper()} PASS: "
                f"{args.capture_dir} <> {args.compare_capture}"
            )
        else:
            print(f"{PASS_PREFIX}: {args.capture_dir}")
        return 0
    except (ArtifactError, OSError) as error:
        print(f"RUNTIME VNEXT CHECKPOINT ARTIFACT REJECT: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
