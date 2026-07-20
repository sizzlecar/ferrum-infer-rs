#!/usr/bin/env python3
"""Validate raw terminal-fence checkpoint artifacts emitted by vNext."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any


PASS_PREFIX = "RUNTIME VNEXT CHECKPOINT ARTIFACT PASS"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ELEMENT_FORMATS = {"f16": ("<e", 2), "f32": ("<f", 4), "u32": ("<I", 4)}

PLAN_FIELDS = frozenset(
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
    plan = exact_object(load_json(plan_path), PLAN_FIELDS, "plan")
    require(plan["schema_version"] == 1, "plan.schema_version must be 1")
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
    checkpoints_raw = plan["checkpoints"]
    require(
        isinstance(checkpoints_raw, list) and checkpoints_raw,
        "plan.checkpoints must be non-empty",
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

    wave_paths = sorted(capture_dir.glob("wave-*.json"))
    require(len(wave_paths) == maximum_waves, "captured wave count differs from plan maximum")
    summaries: list[dict[str, Any]] = []
    referenced_raw: set[str] = set()
    for expected_index, wave_path in enumerate(wave_paths):
        require(
            wave_path.is_file() and not wave_path.is_symlink(),
            "wave manifest must be a real file",
        )
        wave = exact_object(load_json(wave_path), WAVE_FIELDS, f"wave[{expected_index}]")
        require(wave["schema_version"] == 1, "wave.schema_version must be 1")
        require(wave["capture_index"] == expected_index, "wave.capture_index is not contiguous")
        require(
            wave_path.name == f"wave-{expected_index:04}.json",
            "wave filename is not canonical",
        )
        for field in IDENTITY_FIELDS:
            require(wave[field] == plan[field], f"wave.{field} differs from plan")
        require(wave["wave_kind"] == "prefill", "wave.wave_kind must be prefill")
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
                and full_tokens == fit_tokens
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
                    "raw_file": raw_name,
                    "raw_bytes": raw_bytes,
                    "raw_sha256": digest,
                    "element_type": element_type,
                    "stats": tensor_stats(raw_path, element_type, element_count),
                }
            )
        require(record_keys == sorted(set(record_keys)), "wave records must be unique and sorted")
        summaries.append(
            {
                "capture_index": expected_index,
                "participant_count": participant_count,
                "records": records_summary,
            }
        )

    actual_raw = {path.name for path in capture_dir.glob("*.bin")}
    require(actual_raw == referenced_raw, "raw file set differs from manifest references")
    return {
        "schema_version": 1,
        "status": "pass",
        "capture_dir": str(capture_dir.resolve()),
        "model_id": model_id,
        "plan_id": plan["plan_id"],
        "plan_hash": plan_hash,
        "family_fingerprint": plan["family_fingerprint"],
        "program_fingerprint": plan["program_fingerprint"],
        "run_id": plan["run_id"],
        "checkpoint_values": value_ids,
        "wave_count": len(summaries),
        "waves": summaries,
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
            "schema_version": 1,
            **identity,
            "maximum_prefill_waves": 1,
            "checkpoints": [checkpoint],
        }
        wave = {
            "schema_version": 1,
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
                        "fit_input_tokens": 2,
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
        (capture / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        (capture / "wave-0000.json").write_text(json.dumps(wave), encoding="utf-8")
        validate_artifact(capture, "model.test", ["value.test"])
        (capture / raw_name).write_bytes(raw + b"bad")
        try:
            validate_artifact(capture, "model.test", ["value.test"])
        except ArtifactError as error:
            require("raw_bytes differs" in str(error), "self-test rejected the wrong mutation")
        else:
            raise ArtifactError("self-test accepted a mutated raw tensor")
    print("RUNTIME VNEXT CHECKPOINT ARTIFACT SELF-TEST PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", nargs="?", type=Path)
    parser.add_argument("--expected-model-id")
    parser.add_argument("--expected-value", action="append", default=[])
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        require(args.capture_dir is not None, "capture_dir is required")
        summary = validate_artifact(
            args.capture_dir,
            args.expected_model_id,
            args.expected_value,
        )
        if args.summary is not None:
            write_summary(args.summary, summary)
        print(f"{PASS_PREFIX}: {args.capture_dir}")
        return 0
    except (ArtifactError, OSError) as error:
        print(f"RUNTIME VNEXT CHECKPOINT ARTIFACT REJECT: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
