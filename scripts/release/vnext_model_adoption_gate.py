#!/usr/bin/env python3
"""Thin receipt validator for the sequential vNext model-adoption goal.

This command does not run builds, CUDA kernels, product scenarios, or benchmarks.
It validates the four versioned receipts produced by those jobs, verifies every
declared file digest, applies the fixed M0-M6 thresholds, and prints exactly one
checkpoint terminal line for a valid package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "scripts/release/schemas/vnext_model_adoption"
SCHEMA_VERSION = 1
VALIDATOR_VERSION = "1.0.0"
RECEIPT_SCHEMAS = {
    "model-lock.json": "model-lock-v1.schema.json",
    "validation.json": "validation-v1.schema.json",
    "product.json": "product-v1.schema.json",
    "manifest.json": "manifest-v1.schema.json",
}
CHECKPOINTS = {
    "qwen38-27b-fp8": (
        "Qwen/Qwen3.8-27B-FP8",
        "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
    ),
    "qwen36-27b-fp8": (
        "Qwen/Qwen3.6-27B-FP8",
        "e89b16ebf1988b3d6befa7de50abc2d76f26eb09",
    ),
    "qwen36-35b-a3b-fp8": (
        "Qwen/Qwen3.6-35B-A3B-FP8",
        "95a723d08a9490559dae23d0cff1d9466213d989",
    ),
    "gpt-oss-20b-mxfp4": (
        "openai/gpt-oss-20b",
        "6cee5e81ee83917806bbde320786a8fb61efebee",
    ),
    "gemma4-12b-w4a16-ct": (
        "google/gemma-4-12B-it-qat-w4a16-ct",
        "1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee",
    ),
}
STAGE_SECTIONS = {
    "M0": ("model-lock.json", "source_lock"),
    "M1": ("validation.json", "fail_closed"),
    "M2": ("validation.json", "local_path"),
    "M3": ("validation.json", "numeric_approval"),
    "M4": ("product.json", "product_checks"),
    "M5": ("product.json", "usability"),
    "M6": ("validation.json", "architecture_audit"),
}
ARCHITECTURE_CHECKS = {
    "registry_family",
    "typed_quant_layout",
    "plan_catalog_provider",
    "shared_run_serve_identity",
    "no_bypass_hidden_env_fallback",
}
AFFECTED_GROUPS = {
    "family_source",
    "weight_materializer",
    "plan_provider",
    "run_serve_shared_identity",
}
PRODUCT_ERROR_KEYS = {
    "http_500",
    "panic",
    "oom",
    "cuda_error",
    "invalid_utf8",
    "raw_control_or_special_token",
}
FALLBACK_KEYS = {"silent", "dense", "legacy"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SECRET_KEY = re.compile(r"(?:api[_-]?key|token|password|secret|authorization)", re.I)


class GateError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GateError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GateError(f"invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def as_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def as_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def non_empty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be non-empty")
    return value


def finite_number(value: Any, label: str) -> float:
    require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} must be numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} must be finite")
    return result


def positive_int(value: Any, label: str) -> int:
    require(
        not isinstance(value, bool) and isinstance(value, int) and value > 0,
        f"{label} must be a positive integer",
    )
    return value


def parse_time(value: Any, label: str) -> datetime:
    raw = non_empty_string(value, label)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GateError(f"{label} must be ISO-8601 date-time: {raw!r}") from exc
    require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return parsed


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return not isinstance(value, bool) and isinstance(value, int)
    if expected == "number":
        return not isinstance(value, bool) and isinstance(value, (int, float))
    raise GateError(f"validator does not support JSON Schema type {expected!r}")


def _schema_pointer(document: Any, fragment: str, label: str) -> Any:
    if not fragment:
        return document
    require(fragment.startswith("/"), f"unsupported schema fragment in {label}: #{fragment}")
    current = document
    for raw_part in fragment[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        require(isinstance(current, dict) and part in current, f"missing schema pointer {label}")
        current = current[part]
    return current


def _resolve_schema_ref(ref: str, current_schema: Path) -> tuple[Any, Path]:
    file_part, _, fragment = ref.partition("#")
    target = current_schema if not file_part else (current_schema.parent / file_part).resolve()
    require(target.parent == SCHEMA_DIR.resolve(), f"schema ref escapes schema directory: {ref}")
    document = load_json(target)
    return _schema_pointer(document, fragment, ref), target


def _validate_schema(value: Any, schema: Any, label: str, schema_path: Path) -> None:
    schema_obj = as_object(schema, f"schema for {label}")
    if "$ref" in schema_obj:
        target, target_path = _resolve_schema_ref(
            non_empty_string(schema_obj["$ref"], f"schema ref for {label}"), schema_path
        )
        _validate_schema(value, target, label, target_path)
    for index, member in enumerate(schema_obj.get("allOf", [])):
        _validate_schema(value, member, f"{label}.allOf[{index}]", schema_path)
    if "const" in schema_obj:
        require(value == schema_obj["const"], f"{label} must equal {schema_obj['const']!r}")
    if "enum" in schema_obj:
        require(value in schema_obj["enum"], f"{label} must be one of {schema_obj['enum']!r}")
    if "type" in schema_obj:
        raw_types = schema_obj["type"]
        types = [raw_types] if isinstance(raw_types, str) else as_list(raw_types, "schema types")
        require(
            any(_json_type_matches(value, expected) for expected in types),
            f"{label} must have JSON type {types!r}",
        )
    if isinstance(value, dict):
        required = schema_obj.get("required", [])
        for key in required:
            require(key in value, f"{label} missing required property {key!r}")
        properties = schema_obj.get("properties", {})
        for key, child_schema in properties.items():
            if key in value:
                _validate_schema(value[key], child_schema, f"{label}.{key}", schema_path)
        additional = schema_obj.get("additionalProperties", True)
        unknown = set(value) - set(properties)
        if additional is False:
            require(not unknown, f"{label} has unknown properties: {sorted(unknown)}")
        elif isinstance(additional, dict):
            for key in unknown:
                _validate_schema(value[key], additional, f"{label}.{key}", schema_path)
    if isinstance(value, list):
        if "minItems" in schema_obj:
            require(len(value) >= schema_obj["minItems"], f"{label} has too few items")
        if "maxItems" in schema_obj:
            require(len(value) <= schema_obj["maxItems"], f"{label} has too many items")
        if schema_obj.get("uniqueItems"):
            rendered = [json.dumps(item, sort_keys=True) for item in value]
            require(len(rendered) == len(set(rendered)), f"{label} items must be unique")
        if "items" in schema_obj:
            for index, item in enumerate(value):
                _validate_schema(item, schema_obj["items"], f"{label}[{index}]", schema_path)
    if isinstance(value, str):
        if "minLength" in schema_obj:
            require(len(value) >= schema_obj["minLength"], f"{label} is too short")
        if "pattern" in schema_obj:
            require(re.search(schema_obj["pattern"], value) is not None, f"{label} pattern mismatch")
        if schema_obj.get("format") == "date-time":
            parse_time(value, label)
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        if "minimum" in schema_obj:
            require(value >= schema_obj["minimum"], f"{label} is below minimum")
        if "maximum" in schema_obj:
            require(value <= schema_obj["maximum"], f"{label} exceeds maximum")
        if "exclusiveMinimum" in schema_obj:
            require(value > schema_obj["exclusiveMinimum"], f"{label} is below exclusive minimum")


def validate_schema_files() -> None:
    expected = {"common-envelope-v1.schema.json", *RECEIPT_SCHEMAS.values()}
    found = {path.name for path in SCHEMA_DIR.glob("*.schema.json")}
    require(found == expected, f"schema file set mismatch: expected={sorted(expected)} found={sorted(found)}")
    for name in sorted(expected):
        path = SCHEMA_DIR / name
        schema = as_object(load_json(path), str(path))
        require(
            schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
            f"{path} must declare JSON Schema draft 2020-12",
        )
        require(schema.get("$id") == name, f"{path} $id must equal its filename")


def resolve_artifact(out_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else out_dir / path


def verify_reference(reference: Any, label: str, out_dir: Path) -> Path:
    ref = as_object(reference, label)
    path = resolve_artifact(out_dir, non_empty_string(ref.get("path"), f"{label}.path"))
    require(path.is_file(), f"{label} referenced file missing: {path}")
    size = ref.get("size_bytes")
    require(
        not isinstance(size, bool) and isinstance(size, int) and size >= 0,
        f"{label}.size_bytes must be a non-negative integer",
    )
    require(path.stat().st_size == size, f"{label} size mismatch for {path}")
    digest = non_empty_string(ref.get("sha256"), f"{label}.sha256")
    require(HEX64.fullmatch(digest) is not None, f"{label}.sha256 must be lowercase SHA256")
    require(sha256_file(path) == digest, f"{label} SHA256 mismatch for {path}")
    return path


def reference_for(path: Path, rendered_path: str | None = None) -> dict[str, Any]:
    return {
        "path": rendered_path if rendered_path is not None else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_envelope(receipt: dict[str, Any], label: str, out_dir: Path, final_status: str) -> None:
    timing = as_object(receipt["timing"], f"{label}.timing")
    started = parse_time(timing["started_at"], f"{label}.timing.started_at")
    finished = parse_time(timing["finished_at"], f"{label}.timing.finished_at")
    deadline = parse_time(timing["hard_deadline_at"], f"{label}.timing.hard_deadline_at")
    require(finished >= started, f"{label} finished before it started")
    require(deadline >= started, f"{label} hard deadline precedes start")
    duration = finite_number(timing["duration_seconds"], f"{label}.timing.duration_seconds")
    observed = (finished - started).total_seconds()
    require(abs(duration - observed) <= max(5.0, observed * 0.1), f"{label} duration disagrees with timestamps")
    if final_status == "PASS":
        require(finished <= deadline, f"{label} exceeded its declared hard deadline")
    candidate = as_object(receipt["candidate"], f"{label}.candidate")
    dirty_status = as_list(candidate["dirty_status"], f"{label}.candidate.dirty_status")
    if candidate["dirty"]:
        require(bool(dirty_status), f"{label} dirty candidate must list dirty status")
    else:
        require(not dirty_status, f"{label} clean candidate must have empty dirty status")
    environment = as_object(receipt["sanitized_environment"], f"{label}.sanitized_environment")
    for key, value in environment.items():
        require(isinstance(value, str), f"{label}.sanitized_environment.{key} must be a string")
        if SECRET_KEY.search(key):
            require(value == "<redacted>", f"{label} secret-like environment field {key!r} is not redacted")
        require(not re.search(r"Bearer\s+\S+", value, re.I), f"{label} environment contains bearer material")
    argv = as_list(receipt["argv"], f"{label}.argv")
    require(all(isinstance(arg, str) and arg for arg in argv), f"{label}.argv contains an empty argument")
    for index, reference in enumerate(receipt["references"]):
        verify_reference(reference, f"{label}.references[{index}]", out_dir)


def validate_checkpoint_identity(receipt: dict[str, Any], checkpoint_id: str, label: str) -> None:
    expected_repo, expected_revision = CHECKPOINTS[checkpoint_id]
    checkpoint = as_object(receipt["checkpoint"], f"{label}.checkpoint")
    require(checkpoint["id"] == checkpoint_id, f"{label} checkpoint id mismatch")
    require(checkpoint["repository"] == expected_repo, f"{label} checkpoint repository mismatch")
    require(checkpoint["revision"] == expected_revision, f"{label} checkpoint revision mismatch")


def stage_results(receipts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for stage, (filename, _) in STAGE_SECTIONS.items():
        milestones = as_object(receipts[filename]["milestones"], f"{filename}.milestones")
        result[stage] = as_object(milestones[stage], f"{filename}.milestones.{stage}")
    return result


def validate_stage_states(receipts: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> None:
    final_status = manifest["final_status"]
    stages = stage_results(receipts)
    for stage, result in stages.items():
        status = result["status"]
        _, section_name = STAGE_SECTIONS[stage]
        filename, _ = STAGE_SECTIONS[stage]
        section = receipts[filename][section_name]
        if status == "pass":
            require("reason_code" not in result and "reason" not in result, f"{stage} pass has a reason")
            require(isinstance(section, dict), f"{stage} pass requires {filename}.{section_name}")
        else:
            code = non_empty_string(result.get("reason_code"), f"{stage}.reason_code")
            non_empty_string(result.get("reason"), f"{stage}.reason")
            require(code.strip() == code, f"{stage}.reason_code must be normalized")
            if status == "not_run":
                require(section is None, f"{stage} not_run requires a null {section_name}")
    terminal_reason = manifest["terminal_reason"]
    if final_status == "PASS":
        require(terminal_reason is None, "PASS manifest terminal_reason must be null")
        require(all(item["status"] == "pass" for item in stages.values()), "PASS requires M0-M6 pass")
        return
    reason = as_object(terminal_reason, "manifest.terminal_reason")
    require(set(reason) == {"code", "detail"}, "terminal_reason must contain only code and detail")
    reason_code = non_empty_string(reason["code"], "manifest.terminal_reason.code")
    reason_detail = non_empty_string(reason["detail"], "manifest.terminal_reason.detail")
    require(any(item["status"] != "pass" for item in stages.values()), f"{final_status} needs a non-pass stage")
    saw_non_pass = False
    for stage, result in stages.items():
        if saw_non_pass:
            require(result["status"] != "pass", f"{stage} cannot pass after an earlier non-pass stage")
        if result["status"] != "pass":
            saw_non_pass = True
            require(result["reason_code"] == reason_code, f"{stage} does not use the terminal reason code")
            require(result["reason"] == reason_detail, f"{stage} does not use the unique terminal reason")


def required_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(value))
    require(not missing, f"{label} missing fields: {missing}")


def validate_sha(value: Any, label: str) -> str:
    digest = non_empty_string(value, label)
    require(HEX64.fullmatch(digest) is not None, f"{label} must be lowercase SHA256")
    return digest


def validate_m0(source_lock: Any, checkpoint_id: str, *, require_covered: bool) -> dict[str, Any]:
    lock = as_object(source_lock, "model-lock.json.source_lock")
    required_keys(
        lock,
        {
            "identity",
            "lock_checks",
            "files",
            "tensors",
            "partition_counts",
            "expected_quant_tensors",
            "expected_operations",
            "coverage_matrix",
            "quality_vector",
            "digests",
            "memory_estimate",
        },
        "source_lock",
    )
    identity = as_object(lock["identity"], "source_lock.identity")
    repo, revision = CHECKPOINTS[checkpoint_id]
    require(identity.get("repository") == repo, "source_lock identity repository mismatch")
    require(identity.get("revision") == revision, "source_lock identity revision mismatch")
    non_empty_string(identity.get("license"), "source_lock.identity.license")
    non_empty_string(identity.get("architecture"), "source_lock.identity.architecture")
    checks = as_object(lock["lock_checks"], "source_lock.lock_checks")
    expected_checks = {"config", "tokenizer", "template", "index", "shards"}
    require(set(checks) == expected_checks, "source_lock lock_checks set mismatch")
    require(all(checks.values()), "source_lock config/tokenizer/template/index/shards must all be locked")
    files = as_list(lock["files"], "source_lock.files")
    require(bool(files), "source_lock.files must not be empty")
    file_kinds: set[str] = set()
    file_paths: set[str] = set()
    for index, raw in enumerate(files):
        item = as_object(raw, f"source_lock.files[{index}]")
        required_keys(item, {"path", "kind", "size_bytes", "sha256"}, f"source_lock.files[{index}]")
        path = non_empty_string(item["path"], f"source_lock.files[{index}].path")
        require(path not in file_paths, f"duplicate checkpoint file {path}")
        file_paths.add(path)
        file_kinds.add(non_empty_string(item["kind"], f"source_lock.files[{index}].kind"))
        require(
            not isinstance(item["size_bytes"], bool)
            and isinstance(item["size_bytes"], int)
            and item["size_bytes"] >= 0,
            f"source_lock.files[{index}].size_bytes is invalid",
        )
        validate_sha(item["sha256"], f"source_lock.files[{index}].sha256")
    require(
        {"config", "tokenizer", "template", "index", "shard"}.issubset(file_kinds),
        "source_lock files must cover config/tokenizer/template/index/shard",
    )
    tensors = as_list(lock["tensors"], "source_lock.tensors")
    require(bool(tensors), "source_lock.tensors must not be empty")
    tensor_names: set[str] = set()
    disposition_counts = {
        "execution_eligible": 0,
        "typed_non_executed": 0,
        "rejected": 0,
    }
    execution_quant: set[str] = set()
    tensor_by_name: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(tensors):
        tensor = as_object(raw, f"source_lock.tensors[{index}]")
        required_keys(
            tensor,
            {"name", "dtype", "shape", "disposition", "quantized"},
            f"source_lock.tensors[{index}]",
        )
        name = non_empty_string(tensor["name"], f"source_lock.tensors[{index}].name")
        require(name not in tensor_names, f"duplicate tensor inventory entry {name}")
        tensor_names.add(name)
        tensor_by_name[name] = tensor
        non_empty_string(tensor["dtype"], f"source_lock.tensors[{index}].dtype")
        shape = as_list(tensor["shape"], f"source_lock.tensors[{index}].shape")
        require(bool(shape), f"source_lock.tensors[{index}].shape must not be empty")
        for dim_index, dim in enumerate(shape):
            positive_int(dim, f"source_lock.tensors[{index}].shape[{dim_index}]")
        disposition = tensor["disposition"]
        require(disposition in disposition_counts, f"unknown tensor disposition {disposition!r}")
        disposition_counts[disposition] += 1
        if disposition != "execution_eligible":
            non_empty_string(
                tensor.get("classification_reason"),
                f"source_lock.tensors[{index}].classification_reason",
            )
        require(isinstance(tensor["quantized"], bool), f"{name}.quantized must be boolean")
        if tensor["quantized"]:
            layout = as_object(tensor.get("source_layout"), f"{name}.source_layout")
            non_empty_string(layout.get("format"), f"{name}.source_layout.format")
            as_list(layout.get("sidecars"), f"{name}.source_layout.sidecars")
        if disposition == "execution_eligible" and tensor["quantized"]:
            execution_quant.add(name)
    for name, tensor in tensor_by_name.items():
        if not tensor["quantized"]:
            continue
        layout = as_object(tensor["source_layout"], f"{name}.source_layout")
        sidecars = as_list(layout["sidecars"], f"{name}.source_layout.sidecars")
        sidecar_names: set[str] = set()
        for index, raw in enumerate(sidecars):
            sidecar = as_object(raw, f"{name}.source_layout.sidecars[{index}]")
            required_keys(sidecar, {"role", "tensor_name", "dtype", "shape"}, f"{name} sidecar")
            non_empty_string(sidecar["role"], f"{name} sidecar role")
            sidecar_name = non_empty_string(sidecar["tensor_name"], f"{name} sidecar tensor_name")
            require(sidecar_name not in sidecar_names, f"{name} has a duplicate sidecar {sidecar_name}")
            sidecar_names.add(sidecar_name)
            require(sidecar_name in tensor_by_name, f"{name} sidecar {sidecar_name} is absent from inventory")
            inventory_sidecar = tensor_by_name[sidecar_name]
            require(sidecar["dtype"] == inventory_sidecar["dtype"], f"{name} sidecar dtype mismatch")
            require(sidecar["shape"] == inventory_sidecar["shape"], f"{name} sidecar shape mismatch")
            require(
                inventory_sidecar["disposition"] == tensor["disposition"],
                f"{name} sidecar disposition mismatch",
            )
        if checkpoint_id.startswith("qwen3"):
            require(layout["format"] == "fp8_e4m3_block", f"{name} must use fp8_e4m3_block")
            require(layout.get("block_shape") == [128, 128], f"{name} block shape must be 128x128")
            require(len(sidecars) == 1, f"{name} must have exactly one scale_inv sidecar")
            sidecar = sidecars[0]
            require(sidecar["role"] == "scale_inv", f"{name} sidecar role must be scale_inv")
            require(sidecar["dtype"] == "BF16", f"{name} scale_inv dtype must be BF16")
            expected_scale_shape = [
                (tensor["shape"][0] + 127) // 128,
                (tensor["shape"][1] + 127) // 128,
            ]
            require(sidecar["shape"] == expected_scale_shape, f"{name} scale_inv block grid mismatch")
    counts = as_object(lock["partition_counts"], "source_lock.partition_counts")
    expected_count_keys = {*disposition_counts, "unknown", "total"}
    require(set(counts) == expected_count_keys, "partition_counts fields mismatch")
    for disposition, observed in disposition_counts.items():
        require(counts[disposition] == observed, f"partition count mismatch for {disposition}")
    require(counts["unknown"] == 0, "unknown tensor count must be zero")
    require(counts["total"] == len(tensors), "tensor total does not match inventory")
    expected_tensors = as_list(lock["expected_quant_tensors"], "source_lock.expected_quant_tensors")
    require(all(isinstance(item, str) and item for item in expected_tensors), "invalid quant tensor name")
    require(len(expected_tensors) == len(set(expected_tensors)), "expected quant tensors are duplicated")
    require(set(expected_tensors) == execution_quant, "expected quant tensor set does not match inventory")
    operations = as_list(lock["expected_operations"], "source_lock.expected_operations")
    require(bool(operations), "expected_operations must not be empty")
    operation_ids: set[str] = set()
    operation_tensor_union: set[str] = set()
    for index, raw in enumerate(operations):
        operation = as_object(raw, f"source_lock.expected_operations[{index}]")
        required_keys(operation, {"operation_id", "tensor_names"}, "expected operation")
        operation_id = non_empty_string(operation["operation_id"], "expected operation id")
        require(operation_id not in operation_ids, f"duplicate expected operation {operation_id}")
        operation_ids.add(operation_id)
        names = as_list(operation["tensor_names"], f"expected operation {operation_id}.tensor_names")
        require(bool(names), f"expected operation {operation_id} has no tensors")
        require(set(names).issubset(execution_quant), f"expected operation {operation_id} has unknown tensors")
        operation_tensor_union.update(names)
    require(operation_tensor_union == execution_quant, "expected operation tensor union is incomplete")
    coverage = as_list(lock["coverage_matrix"], "source_lock.coverage_matrix")
    covered_ops: set[str] = set()
    coverage_ops: list[str] = []
    coverage_pairs: set[tuple[str, str]] = set()
    for index, raw in enumerate(coverage):
        cell = as_object(raw, f"source_lock.coverage_matrix[{index}]")
        required_keys(cell, {"operation_id", "provider_id", "covered"}, "coverage cell")
        operation_id = non_empty_string(cell["operation_id"], "coverage operation id")
        require(operation_id in operation_ids, f"coverage has unexpected operation {operation_id}")
        coverage_ops.append(operation_id)
        provider_id = non_empty_string(cell["provider_id"], "coverage provider id")
        pair = (operation_id, provider_id)
        require(pair not in coverage_pairs, f"duplicate coverage cell {pair}")
        coverage_pairs.add(pair)
        require(isinstance(cell["covered"], bool), "coverage covered must be boolean")
        if cell["covered"]:
            covered_ops.add(operation_id)
    require(set(coverage_ops) == operation_ids, "coverage matrix is incomplete")
    if require_covered:
        require(covered_ops == operation_ids, "PASS requires provider coverage for every expected operation")
    quality = as_object(lock["quality_vector"], "source_lock.quality_vector")
    for key in ["generator_semantics", "input_semantics", "reference_semantics"]:
        non_empty_string(quality.get(key), f"source_lock.quality_vector.{key}")
    digests = as_object(lock["digests"], "source_lock.digests")
    digest_names = {
        "checkpoint_content_digest",
        "source_schema_fingerprint",
        "execution_contract_fingerprint",
        "quality_vector_digest",
    }
    require(set(digests) == digest_names, "source_lock digest set mismatch")
    for name in digest_names:
        validate_sha(digests[name], f"source_lock.digests.{name}")
    memory = as_object(lock["memory_estimate"], "source_lock.memory_estimate")
    positive_int(memory.get("peak_host_bytes"), "source_lock.memory_estimate.peak_host_bytes")
    positive_int(memory.get("peak_device_bytes"), "source_lock.memory_estimate.peak_device_bytes")
    non_empty_string(memory.get("rationale"), "source_lock.memory_estimate.rationale")
    denominator = canonical_sha256(
        {"operations": sorted(operation_ids), "quant_tensors": sorted(execution_quant)}
    )
    return {
        "digests": digests,
        "denominator": denominator,
        "expected_count": len(operation_ids) + len(execution_quant),
        "reference_semantics": quality["reference_semantics"],
    }


def validate_m1(value: Any) -> None:
    section = as_object(value, "validation.json.fail_closed")
    cases = as_list(section.get("cases"), "validation.json.fail_closed.cases")
    require(len(cases) == 2, "M1 requires exactly two bad-contract cases")
    kinds: set[str] = set()
    ids: set[str] = set()
    for index, raw in enumerate(cases):
        case = as_object(raw, f"M1 case[{index}]")
        required_keys(
            case,
            {"case_id", "kind", "typed_error_code", "rejected_before_gpu_allocation", "gpu_allocations"},
            f"M1 case[{index}]",
        )
        case_id = non_empty_string(case["case_id"], f"M1 case[{index}].case_id")
        require(case_id not in ids, f"duplicate M1 case id {case_id}")
        ids.add(case_id)
        kinds.add(non_empty_string(case["kind"], f"M1 case[{index}].kind"))
        non_empty_string(case["typed_error_code"], f"M1 case[{index}].typed_error_code")
        require(case["rejected_before_gpu_allocation"] is True, "M1 rejection was not pre-GPU")
        require(case["gpu_allocations"] == 0, "M1 bad-contract case allocated GPU memory")
    require(kinds == {"metadata_recipe_mismatch", "tensor_layout_mismatch"}, "M1 case kinds mismatch")


def command_argv(value: Any, label: str) -> list[str]:
    command = as_object(value, label)
    argv = as_list(command.get("argv"), f"{label}.argv")
    require(all(isinstance(arg, str) and arg for arg in argv), f"{label}.argv is invalid")
    require(command.get("exit_code") == 0, f"{label} exit_code must be zero")
    return argv


def validate_m2(value: Any, out_dir: Path) -> None:
    section = as_object(value, "validation.json.local_path")
    require(section.get("candidate_frozen") is True, "M2 candidate_frozen must be true")
    build = command_argv(section.get("release_build"), "M2 release_build")
    expected = [
        "cargo",
        "build",
        "--release",
        "-p",
        "ferrum-cli",
        "--bin",
        "ferrum",
        "--features",
        "cuda,vllm-moe-marlin,vllm-paged-attn-v2",
    ]
    require(build == expected, f"M2 release build argv mismatch: {build!r}")
    affected = as_list(section.get("affected_commands"), "M2 affected_commands")
    require(len(affected) == 4, "M2 requires exactly four affected command groups")
    groups: set[str] = set()
    for index, raw in enumerate(affected):
        item = as_object(raw, f"M2 affected_commands[{index}]")
        group = non_empty_string(item.get("group"), f"M2 affected_commands[{index}].group")
        groups.add(group)
        command_argv(item, f"M2 affected_commands[{index}]")
    require(groups == AFFECTED_GROUPS, "M2 affected command group set mismatch")
    unit = as_object(section.get("unit_gate"), "M2 unit_gate")
    argv = command_argv(unit, "M2 unit_gate")
    joined = " ".join(argv)
    require("scripts/release/run_gate.py unit" in joined, "M2 must use run_gate.py unit")
    pass_line = non_empty_string(unit.get("pass_line"), "M2 unit_gate.pass_line")
    require(pass_line.startswith("FERRUM GATE unit PASS: "), "M2 unit gate PASS line mismatch")
    verify_reference(unit.get("artifact"), "M2 unit_gate.artifact", out_dir)


def validate_numeric_case(case: Any, label: str) -> tuple[tuple[int, int], int]:
    item = as_object(case, label)
    required_keys(
        item,
        {"case_id", "weight_shape", "activation_batch", "relative_l2", "nan_count", "inf_count"},
        label,
    )
    non_empty_string(item["case_id"], f"{label}.case_id")
    shape = as_list(item["weight_shape"], f"{label}.weight_shape")
    require(len(shape) == 2, f"{label}.weight_shape must have two dimensions")
    dims = tuple(positive_int(dim, f"{label}.weight_shape") for dim in shape)
    batch = positive_int(item["activation_batch"], f"{label}.activation_batch")
    relative_l2 = finite_number(item["relative_l2"], f"{label}.relative_l2")
    require(0.0 <= relative_l2 <= 0.05, f"{label}.relative_l2 exceeds 0.05")
    require(item["nan_count"] == 0, f"{label} contains NaN")
    require(item["inf_count"] == 0, f"{label} contains Inf")
    return (dims[0], dims[1]), batch


def validate_m3(value: Any, checkpoint_id: str, m0: dict[str, Any], out_dir: Path) -> None:
    section = as_object(value, "validation.json.numeric_approval")
    required_keys(section, {"mode", "quality_vector_digest", "reference_semantics", "cases", "approval"}, "M3")
    digests = m0["digests"]
    require(section["quality_vector_digest"] == digests["quality_vector_digest"], "M3 quality digest mismatch")
    non_empty_string(section["reference_semantics"], "M3 reference_semantics")
    require(
        section["reference_semantics"] == m0["reference_semantics"],
        "M3 reference semantics differ from the M0 lock",
    )
    cases = as_list(section["cases"], "M3 cases")
    ids: set[str] = set()
    cells: set[tuple[tuple[int, int], int]] = set()
    for index, case in enumerate(cases):
        item = as_object(case, f"M3 cases[{index}]")
        case_id = non_empty_string(item.get("case_id"), f"M3 cases[{index}].case_id")
        require(case_id not in ids, f"duplicate M3 case id {case_id}")
        ids.add(case_id)
        cells.add(validate_numeric_case(item, f"M3 cases[{index}]"))
    mode = section["mode"]
    if mode == "full":
        require(len(cases) == 4 and len(cells) == 4, "M3 full mode requires four distinct cases")
        shapes = {shape for shape, _ in cells}
        batches = {batch for _, batch in cells}
        require(len(shapes) == 2 and len(batches) == 2, "M3 requires 2 shapes x 2 batches")
        require(cells == {(shape, batch) for shape in shapes for batch in batches}, "M3 matrix is incomplete")
    elif mode == "reuse_canary":
        require(
            checkpoint_id in {"qwen36-27b-fp8", "qwen36-35b-a3b-fp8"},
            "M3 reuse_canary is only valid for A2/A3",
        )
        require(len(cases) == 1, "M3 reuse_canary requires exactly one case")
        reuse = as_object(section.get("reuse_evidence"), "M3 reuse_evidence")
        required_keys(
            reuse,
            {
                "checkpoint_id",
                "validation_receipt",
                "source_schema_fingerprint",
                "execution_contract_fingerprint",
                "quality_vector_digest",
            },
            "M3 reuse_evidence",
        )
        require(reuse["checkpoint_id"] in CHECKPOINTS, "M3 reuse checkpoint is unknown")
        require(reuse["checkpoint_id"] != checkpoint_id, "M3 cannot reuse itself")
        reused_path = verify_reference(reuse["validation_receipt"], "M3 reuse validation receipt", out_dir)
        reused = as_object(load_json(reused_path), "M3 reused validation receipt")
        reused_checkpoint = as_object(reused.get("checkpoint"), "M3 reused checkpoint")
        require(reused_checkpoint.get("id") == reuse["checkpoint_id"], "M3 reused checkpoint id mismatch")
        reused_repo, reused_revision = CHECKPOINTS[reuse["checkpoint_id"]]
        require(reused_checkpoint.get("repository") == reused_repo, "M3 reused repository mismatch")
        require(reused_checkpoint.get("revision") == reused_revision, "M3 reused revision mismatch")
        reused_approval = as_object(reused.get("numeric_approval"), "M3 reused numeric approval")
        reused_binding = as_object(reused_approval.get("approval"), "M3 reused approval binding")
        for key in [
            "source_schema_fingerprint",
            "execution_contract_fingerprint",
            "quality_vector_digest",
        ]:
            require(reuse[key] == digests[key], f"M3 reuse {key} differs from current lock")
            require(reused_binding.get(key) == digests[key], f"M3 reused receipt {key} mismatch")
    else:
        raise GateError(f"M3 mode must be full or reuse_canary, got {mode!r}")
    approval = as_object(section["approval"], "M3 approval")
    required_keys(
        approval,
        {
            "approved",
            "materializer_id",
            "materializer_version",
            "implementation_fingerprint",
            "source_schema_fingerprint",
            "execution_contract_fingerprint",
            "quality_vector_digest",
        },
        "M3 approval",
    )
    require(approval["approved"] is True, "M3 typed approval is not approved")
    for key in ["materializer_id", "materializer_version"]:
        non_empty_string(approval[key], f"M3 approval.{key}")
    for key in [
        "implementation_fingerprint",
        "source_schema_fingerprint",
        "execution_contract_fingerprint",
        "quality_vector_digest",
    ]:
        validate_sha(approval[key], f"M3 approval.{key}")
    require(
        approval["source_schema_fingerprint"] == digests["source_schema_fingerprint"],
        "M3 approval source schema mismatch",
    )
    require(
        approval["execution_contract_fingerprint"] == digests["execution_contract_fingerprint"],
        "M3 approval execution contract mismatch",
    )
    require(
        approval["quality_vector_digest"] == digests["quality_vector_digest"],
        "M3 approval quality vector mismatch",
    )


def argv_contains_subcommand(argv: Any, subcommand: str, label: str) -> list[str]:
    args = as_list(argv, label)
    require(all(isinstance(arg, str) and arg for arg in args), f"{label} is invalid")
    require(subcommand in args, f"{label} does not contain {subcommand!r}")
    return args


def flag_value(argv: list[str], flag: str, label: str) -> str:
    require(flag in argv, f"{label} missing {flag}")
    index = argv.index(flag)
    require(index + 1 < len(argv), f"{label} missing value for {flag}")
    return argv[index + 1]


def validate_hardware(value: Any) -> None:
    hardware = as_object(value, "product.json.hardware")
    required_keys(hardware, {"gpu_count", "gpus", "driver_version", "cuda_runtime"}, "product hardware")
    require(hardware["gpu_count"] == 1, "product evidence must use exactly one GPU")
    gpus = as_list(hardware["gpus"], "product hardware.gpus")
    require(len(gpus) == 1, "product hardware must identify exactly one GPU")
    gpu = as_object(gpus[0], "product hardware.gpus[0]")
    non_empty_string(gpu.get("name"), "product GPU name")
    positive_int(gpu.get("memory_bytes"), "product GPU memory_bytes")
    non_empty_string(hardware["driver_version"], "product driver_version")
    non_empty_string(hardware["cuda_runtime"], "product cuda_runtime")


def validate_m4(value: Any, checkpoint_id: str) -> None:
    section = as_object(value, "product.json.product_checks")
    required_keys(
        section,
        {
            "load_to_ready_seconds",
            "shared_identity",
            "run",
            "serve_argv",
            "serve_non_stream",
            "serve_stream",
            "stability_c2",
            "errors",
        },
        "M4",
    )
    load_time = finite_number(section["load_to_ready_seconds"], "M4 load_to_ready_seconds")
    require(0 <= load_time <= 600, "M4 load-to-ready exceeds 600 seconds")
    shared_identity = as_object(section["shared_identity"], "M4 shared_identity")
    require(set(shared_identity) == {"run", "serve"}, "M4 shared identity must contain run and serve")
    run_identity = as_object(shared_identity["run"], "M4 shared_identity.run")
    serve_identity = as_object(shared_identity["serve"], "M4 shared_identity.serve")
    identity_fields = {
        "prepared_family_id",
        "plan_fingerprint",
        "weight_decision_fingerprint",
        "tokenizer_digest",
        "chat_template_digest",
    }
    require(set(run_identity) == identity_fields, "M4 run identity field set mismatch")
    require(set(serve_identity) == identity_fields, "M4 serve identity field set mismatch")
    for field in identity_fields:
        non_empty_string(run_identity[field], f"M4 run identity {field}")
    require(run_identity == serve_identity, "M4 run and serve identities differ")
    run = as_object(section["run"], "M4 run")
    run_argv = argv_contains_subcommand(run.get("argv"), "run", "M4 run.argv")
    require(run.get("exit_code") == 0, "M4 ferrum run failed")
    require(run.get("assistant_nonempty") is True, "M4 ferrum run assistant is empty")
    require(run.get("marker_matched") is True, "M4 ferrum run marker did not match")
    serve_argv = argv_contains_subcommand(section["serve_argv"], "serve", "M4 serve_argv")
    if checkpoint_id.startswith("qwen3"):
        require("--disable-thinking" in run_argv, "Qwen M4 run must use --disable-thinking")
    non_stream = as_object(section["serve_non_stream"], "M4 serve_non_stream")
    require(non_stream.get("http_status") == 200, "M4 non-stream HTTP status must be 200")
    require(non_stream.get("json_parseable") is True, "M4 non-stream response is not JSON")
    require(non_stream.get("assistant_nonempty") is True, "M4 non-stream assistant is empty")
    stream = as_object(section["serve_stream"], "M4 serve_stream")
    require(stream.get("http_status") == 200, "M4 stream HTTP status must be 200")
    require(stream.get("done_count") == 1, "M4 stream must have exactly one [DONE]")
    require(stream.get("usage_chunk_count") == 1, "M4 stream must have exactly one usage chunk")
    positive_int(stream.get("output_tokens"), "M4 stream output_tokens")
    if checkpoint_id.startswith("qwen3"):
        for label, request in [("non-stream", non_stream), ("stream", stream)]:
            kwargs = as_object(request.get("chat_template_kwargs"), f"M4 {label} chat_template_kwargs")
            require(
                kwargs.get("enable_thinking") is False,
                f"Qwen M4 {label} request must set chat_template_kwargs.enable_thinking=false",
            )
    stability = as_object(section["stability_c2"], "M4 stability_c2")
    require(stability.get("concurrency") == 2, "M4 stability concurrency must be 2")
    require(stability.get("requests") == 4, "M4 stability must issue four requests")
    require(
        stability.get("input_tokens_per_request") == 256,
        "M4 stability input length must be 256",
    )
    require(
        stability.get("requested_output_tokens_per_request") == 32,
        "M4 stability requested output length must be 32",
    )
    require(stability.get("successful_requests") == 4, "M4 stability must pass 4/4 requests")
    require(stability.get("min_output_tokens") >= 16, "M4 stability output tokens are below 16")
    errors = as_object(section["errors"], "M4 errors")
    require(set(errors) == PRODUCT_ERROR_KEYS, "M4 error counter set mismatch")
    require(all(value == 0 for value in errors.values()), "M4 product error counters must all be zero")
    require(bool(serve_argv), "M4 serve argv is empty")


def validate_m5(value: Any, checkpoint_id: str, m0: dict[str, Any]) -> None:
    section = as_object(value, "product.json.usability")
    required_keys(
        section,
        {
            "bench_argv",
            "request_count",
            "median_output_throughput_tokens_per_second",
            "p50_ttft_seconds",
            "output_token_count_source",
            "provider_attribution",
            "fallback_counts",
        },
        "M5",
    )
    argv = argv_contains_subcommand(section["bench_argv"], "bench-serve", "M5 bench_argv")
    require("--fail-on-error" in argv, "M5 bench must use --fail-on-error")
    require(flag_value(argv, "--seed", "M5 bench") == "9271", "M5 bench seed must be 9271")
    require(flag_value(argv, "--n-repeats", "M5 bench") == "1", "M5 n-repeats must be 1")
    require(flag_value(argv, "--concurrency", "M5 bench") == "1", "M5 concurrency must be 1")
    require(flag_value(argv, "--num-prompts", "M5 bench") == "3", "M5 num-prompts must be 3")
    require(section["request_count"] == 3, "M5 must measure exactly three requests")
    if checkpoint_id.startswith("qwen3"):
        require(
            flag_value(argv, "--enable-thinking", "M5 bench") == "false",
            "Qwen M5 bench must disable thinking through the typed option",
        )
    throughput = finite_number(
        section["median_output_throughput_tokens_per_second"], "M5 median throughput"
    )
    ttft = finite_number(section["p50_ttft_seconds"], "M5 p50 TTFT")
    require(throughput >= 5.0, "M5 median output throughput is below 5 tok/s")
    require(0 <= ttft <= 60.0, "M5 p50 TTFT exceeds 60 seconds")
    require(section["output_token_count_source"] == "usage", "M5 output tokens must come from usage")
    attribution = as_object(section["provider_attribution"], "M5 provider_attribution")
    required_keys(
        attribution,
        {"expected_item_count", "attributed_item_count", "percent", "denominator_sha256"},
        "M5 provider_attribution",
    )
    require(attribution["expected_item_count"] == m0["expected_count"], "M5 attribution denominator count mismatch")
    require(attribution["attributed_item_count"] == m0["expected_count"], "M5 attribution is not 100%")
    require(finite_number(attribution["percent"], "M5 attribution percent") == 100.0, "M5 attribution percent must be 100")
    require(attribution["denominator_sha256"] == m0["denominator"], "M5 attribution denominator digest mismatch")
    fallbacks = as_object(section["fallback_counts"], "M5 fallback_counts")
    require(set(fallbacks) == FALLBACK_KEYS, "M5 fallback counter set mismatch")
    require(all(value == 0 for value in fallbacks.values()), "M5 fallback counts must all be zero")


def validate_m6(value: Any) -> None:
    section = as_object(value, "validation.json.architecture_audit")
    checklist = as_object(section.get("checklist"), "M6 checklist")
    require(set(checklist) == ARCHITECTURE_CHECKS, "M6 checklist item set mismatch")
    require(all(value is True for value in checklist.values()), "M6 checklist must pass 5/5")
    require(section.get("validator_self_test_passed") is True, "M6 validator self-test was not recorded")
    schemas = as_list(section.get("schemas"), "M6 schemas")
    require(set(schemas) == set(RECEIPT_SCHEMAS.values()), "M6 schema set mismatch")


def expected_terminal_line(status: str, checkpoint_id: str, out_dir: Path) -> str:
    return f"FERRUM VNEXT MODEL ADOPTION {status}: {checkpoint_id} {out_dir}"


def load_receipts(out_dir: Path) -> dict[str, dict[str, Any]]:
    require(out_dir.is_dir(), f"artifact directory does not exist: {out_dir}")
    json_files = {str(path.relative_to(out_dir)) for path in out_dir.rglob("*.json")}
    require(
        json_files == set(RECEIPT_SCHEMAS),
        f"artifact root must contain exactly four receipt JSON files; found={sorted(json_files)}",
    )
    receipts: dict[str, dict[str, Any]] = {}
    for filename, schema_name in RECEIPT_SCHEMAS.items():
        receipt_path = out_dir / filename
        receipt = as_object(load_json(receipt_path), filename)
        _validate_schema(receipt, load_json(SCHEMA_DIR / schema_name), filename, SCHEMA_DIR / schema_name)
        receipts[filename] = receipt
    return receipts


def validate_package(checkpoint_id: str, out_dir: Path, *, write_log: bool = True) -> str:
    require(checkpoint_id in CHECKPOINTS, f"unknown checkpoint id: {checkpoint_id}")
    validate_schema_files()
    receipts = load_receipts(out_dir)
    manifest = receipts["manifest.json"]
    final_status = manifest["final_status"]
    for filename, receipt in receipts.items():
        validate_checkpoint_identity(receipt, checkpoint_id, filename)
        validate_envelope(receipt, filename, out_dir, final_status)
    candidates = [receipt["candidate"] for receipt in receipts.values()]
    require(all(candidate == candidates[0] for candidate in candidates[1:]), "candidate identity differs across receipts")
    for filename in ["model-lock.json", "validation.json", "product.json"]:
        reference = manifest["receipts"][filename]
        require(reference["path"] == filename, f"manifest receipt path must be exactly {filename}")
        verify_reference(reference, f"manifest.receipts.{filename}", out_dir)
    require(manifest["validator_version"] == VALIDATOR_VERSION, "manifest validator_version mismatch")
    expected_line = expected_terminal_line(final_status, checkpoint_id, out_dir)
    require(manifest["terminal_line"] == expected_line, "manifest terminal line mismatch")
    validate_stage_states(receipts, manifest)
    stages = stage_results(receipts)
    m0_result: dict[str, Any] | None = None
    if stages["M0"]["status"] == "pass":
        m0_result = validate_m0(
            receipts["model-lock.json"]["source_lock"],
            checkpoint_id,
            require_covered=final_status == "PASS",
        )
    if stages["M1"]["status"] == "pass":
        validate_m1(receipts["validation.json"]["fail_closed"])
    if stages["M2"]["status"] == "pass":
        validate_m2(receipts["validation.json"]["local_path"], out_dir)
    if stages["M3"]["status"] == "pass":
        require(m0_result is not None, "M3 cannot pass without a valid M0")
        validate_m3(receipts["validation.json"]["numeric_approval"], checkpoint_id, m0_result, out_dir)
    if stages["M4"]["status"] == "pass":
        validate_hardware(receipts["product.json"]["hardware"])
        require(bool(receipts["product.json"]["effective_config"]), "M4 effective_config must not be empty")
        validate_m4(receipts["product.json"]["product_checks"], checkpoint_id)
    if stages["M5"]["status"] == "pass":
        require(m0_result is not None, "M5 cannot pass without a valid M0")
        validate_m5(receipts["product.json"]["usability"], checkpoint_id, m0_result)
    if stages["M6"]["status"] == "pass":
        validate_m6(receipts["validation.json"]["architecture_audit"])
    if final_status == "PASS":
        validation_binary = validate_sha(
            receipts["validation.json"]["binary_sha256"], "validation.json.binary_sha256"
        )
        product_binary = validate_sha(
            receipts["product.json"]["binary_sha256"], "product.json.binary_sha256"
        )
        require(validation_binary == product_binary, "validation/product binary SHA256 mismatch")
    if write_log:
        log_lines = [
            f"validator_version={VALIDATOR_VERSION}",
            f"checkpoint_id={checkpoint_id}",
            f"final_status={final_status}",
        ]
        for filename in RECEIPT_SCHEMAS:
            log_lines.append(f"{filename}.sha256={sha256_file(out_dir / filename)}")
        log_lines.append(expected_line)
        (out_dir / "validator.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return expected_line


def synthetic_envelope(artifact_type: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "checkpoint": {
            "id": "qwen38-27b-fp8",
            "repository": CHECKPOINTS["qwen38-27b-fp8"][0],
            "revision": CHECKPOINTS["qwen38-27b-fp8"][1],
        },
        "candidate": {"git_sha": "1" * 40, "dirty": False, "dirty_status": []},
        "sanitized_environment": {"CUDA_VISIBLE_DEVICES": "0", "HF_TOKEN": "<redacted>"},
        "argv": ["synthetic", artifact_type],
        "timing": {
            "started_at": "2026-08-29T00:00:00+00:00",
            "finished_at": "2026-08-29T00:00:01+00:00",
            "duration_seconds": 1.0,
            "expected_duration_seconds": 10.0,
            "hard_deadline_at": "2026-08-29T00:01:00+00:00",
            "progress_signal": "synthetic receipt bytes",
        },
        "references": [],
    }


def synthetic_pass_documents(out_dir: Path) -> dict[str, dict[str, Any]]:
    unit_manifest = out_dir.parent / f"{out_dir.name}-upstream" / "unit.gate.json"
    write_json(unit_manifest, {"status": "pass"})
    expected_tensors = ["model.layers.0.q_proj.weight", "model.layers.0.o_proj.weight"]
    expected_operations = [
        {"operation_id": "op.q_proj", "tensor_names": [expected_tensors[0]]},
        {"operation_id": "op.o_proj", "tensor_names": [expected_tensors[1]]},
    ]
    denominator = canonical_sha256(
        {"operations": ["op.o_proj", "op.q_proj"], "quant_tensors": sorted(expected_tensors)}
    )
    digests = {
        "checkpoint_content_digest": "a" * 64,
        "source_schema_fingerprint": "b" * 64,
        "execution_contract_fingerprint": "c" * 64,
        "quality_vector_digest": "d" * 64,
    }
    model_lock = {
        **synthetic_envelope("model-lock"),
        "milestones": {"M0": {"status": "pass"}},
        "source_lock": {
            "identity": {
                "repository": CHECKPOINTS["qwen38-27b-fp8"][0],
                "revision": CHECKPOINTS["qwen38-27b-fp8"][1],
                "license": "apache-2.0",
                "architecture": "Qwen3_5ForConditionalGeneration",
            },
            "lock_checks": {
                "config": True,
                "tokenizer": True,
                "template": True,
                "index": True,
                "shards": True,
            },
            "files": [
                {"path": "config.json", "kind": "config", "size_bytes": 1, "sha256": "1" * 64},
                {"path": "tokenizer.json", "kind": "tokenizer", "size_bytes": 1, "sha256": "2" * 64},
                {"path": "template.jinja", "kind": "template", "size_bytes": 1, "sha256": "3" * 64},
                {"path": "model.index.json", "kind": "index", "size_bytes": 1, "sha256": "4" * 64},
                {"path": "model-00001.safetensors", "kind": "shard", "size_bytes": 1, "sha256": "5" * 64},
            ],
            "tensors": [
                {
                    "name": expected_tensors[0],
                    "dtype": "F8_E4M3",
                    "shape": [128, 128],
                    "disposition": "execution_eligible",
                    "quantized": True,
                    "source_layout": {
                        "format": "fp8_e4m3_block",
                        "block_shape": [128, 128],
                        "sidecars": [
                            {
                                "role": "scale_inv",
                                "tensor_name": "model.layers.0.q_proj.weight_scale_inv",
                                "dtype": "BF16",
                                "shape": [1, 1],
                            }
                        ],
                    },
                },
                {
                    "name": expected_tensors[1],
                    "dtype": "F8_E4M3",
                    "shape": [256, 128],
                    "disposition": "execution_eligible",
                    "quantized": True,
                    "source_layout": {
                        "format": "fp8_e4m3_block",
                        "block_shape": [128, 128],
                        "sidecars": [
                            {
                                "role": "scale_inv",
                                "tensor_name": "model.layers.0.o_proj.weight_scale_inv",
                                "dtype": "BF16",
                                "shape": [2, 1],
                            }
                        ],
                    },
                },
                {
                    "name": "model.layers.0.q_proj.weight_scale_inv",
                    "dtype": "BF16",
                    "shape": [1, 1],
                    "disposition": "execution_eligible",
                    "quantized": False,
                },
                {
                    "name": "model.layers.0.o_proj.weight_scale_inv",
                    "dtype": "BF16",
                    "shape": [2, 1],
                    "disposition": "execution_eligible",
                    "quantized": False,
                },
                {
                    "name": "visual.patch_embed.weight",
                    "dtype": "BF16",
                    "shape": [128, 128],
                    "disposition": "typed_non_executed",
                    "quantized": False,
                    "classification_reason": "typed text-only capability excludes vision",
                },
            ],
            "partition_counts": {
                "execution_eligible": 4,
                "typed_non_executed": 1,
                "rejected": 0,
                "unknown": 0,
                "total": 5,
            },
            "expected_quant_tensors": expected_tensors,
            "expected_operations": expected_operations,
            "coverage_matrix": [
                {"operation_id": "op.q_proj", "provider_id": "provider.cuda.fp8", "covered": True},
                {"operation_id": "op.o_proj", "provider_id": "provider.cuda.fp8", "covered": True},
            ],
            "quality_vector": {
                "generator_semantics": "seeded exact source block-FP8 fixtures",
                "input_semantics": "two shapes by two activation batches",
                "reference_semantics": "matmul after source values are decoded with locked scales/layout",
            },
            "digests": digests,
            "memory_estimate": {
                "peak_host_bytes": 40_000_000_000,
                "peak_device_bytes": 35_000_000_000,
                "rationale": "checkpoint bytes plus one transient component and runtime buffers",
            },
        },
    }
    validation = {
        **synthetic_envelope("validation"),
        "binary_sha256": "e" * 64,
        "milestones": {stage: {"status": "pass"} for stage in ["M1", "M2", "M3", "M6"]},
        "fail_closed": {
            "cases": [
                {
                    "case_id": "bad-recipe",
                    "kind": "metadata_recipe_mismatch",
                    "typed_error_code": "quantization.recipe_mismatch",
                    "rejected_before_gpu_allocation": True,
                    "gpu_allocations": 0,
                },
                {
                    "case_id": "bad-layout",
                    "kind": "tensor_layout_mismatch",
                    "typed_error_code": "weight.sidecar_shape_mismatch",
                    "rejected_before_gpu_allocation": True,
                    "gpu_allocations": 0,
                },
            ]
        },
        "local_path": {
            "candidate_frozen": True,
            "release_build": {
                "argv": [
                    "cargo",
                    "build",
                    "--release",
                    "-p",
                    "ferrum-cli",
                    "--bin",
                    "ferrum",
                    "--features",
                    "cuda,vllm-moe-marlin,vllm-paged-attn-v2",
                ],
                "exit_code": 0,
            },
            "affected_commands": [
                {"group": group, "argv": ["cargo", "test", group], "exit_code": 0}
                for group in sorted(AFFECTED_GROUPS)
            ],
            "unit_gate": {
                "argv": ["python3", "scripts/release/run_gate.py", "unit", "--out", "unit"],
                "exit_code": 0,
                "pass_line": "FERRUM GATE unit PASS: unit",
                "artifact": reference_for(unit_manifest),
            },
        },
        "numeric_approval": {
            "mode": "full",
            "quality_vector_digest": digests["quality_vector_digest"],
            "reference_semantics": "matmul after source values are decoded with locked scales/layout",
            "cases": [
                {
                    "case_id": f"shape-{shape[0]}-{shape[1]}-batch-{batch}",
                    "weight_shape": list(shape),
                    "activation_batch": batch,
                    "relative_l2": 0.01,
                    "nan_count": 0,
                    "inf_count": 0,
                }
                for shape in [(128, 128), (256, 128)]
                for batch in [1, 4]
            ],
            "approval": {
                "approved": True,
                "materializer_id": "weight-materializer.cuda.block-fp8",
                "materializer_version": "1.0.0",
                "implementation_fingerprint": "f" * 64,
                "source_schema_fingerprint": digests["source_schema_fingerprint"],
                "execution_contract_fingerprint": digests["execution_contract_fingerprint"],
                "quality_vector_digest": digests["quality_vector_digest"],
            },
        },
        "architecture_audit": {
            "checklist": {key: True for key in sorted(ARCHITECTURE_CHECKS)},
            "validator_self_test_passed": True,
            "schemas": sorted(RECEIPT_SCHEMAS.values()),
        },
    }
    validation["references"] = [reference_for(unit_manifest)]
    product = {
        **synthetic_envelope("product"),
        "binary_sha256": "e" * 64,
        "hardware": {
            "gpu_count": 1,
            "gpus": [{"name": "NVIDIA L40S", "memory_bytes": 48_000_000_000}],
            "driver_version": "synthetic-driver",
            "cuda_runtime": "synthetic-cuda",
        },
        "effective_config": {"backend": "cuda", "text_only": True},
        "milestones": {"M4": {"status": "pass"}, "M5": {"status": "pass"}},
        "product_checks": {
            "load_to_ready_seconds": 500.0,
            "shared_identity": {
                "run": {
                    "prepared_family_id": "family.qwen3_5.hybrid",
                    "plan_fingerprint": "plan-fixture",
                    "weight_decision_fingerprint": "weight-fixture",
                    "tokenizer_digest": "tokenizer-fixture",
                    "chat_template_digest": "template-fixture",
                },
                "serve": {
                    "prepared_family_id": "family.qwen3_5.hybrid",
                    "plan_fingerprint": "plan-fixture",
                    "weight_decision_fingerprint": "weight-fixture",
                    "tokenizer_digest": "tokenizer-fixture",
                    "chat_template_digest": "template-fixture",
                },
            },
            "run": {
                "argv": ["ferrum", "run", "--disable-thinking"],
                "exit_code": 0,
                "assistant_nonempty": True,
                "marker_matched": True,
            },
            "serve_argv": ["ferrum", "serve"],
            "serve_non_stream": {
                "http_status": 200,
                "json_parseable": True,
                "assistant_nonempty": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "serve_stream": {
                "http_status": 200,
                "done_count": 1,
                "usage_chunk_count": 1,
                "output_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            "stability_c2": {
                "concurrency": 2,
                "requests": 4,
                "input_tokens_per_request": 256,
                "requested_output_tokens_per_request": 32,
                "successful_requests": 4,
                "min_output_tokens": 16,
            },
            "errors": {key: 0 for key in sorted(PRODUCT_ERROR_KEYS)},
        },
        "usability": {
            "bench_argv": [
                "ferrum",
                "bench-serve",
                "--fail-on-error",
                "--seed",
                "9271",
                "--n-repeats",
                "1",
                "--concurrency",
                "1",
                "--num-prompts",
                "3",
                "--enable-thinking",
                "false",
            ],
            "request_count": 3,
            "median_output_throughput_tokens_per_second": 5.0,
            "p50_ttft_seconds": 60.0,
            "output_token_count_source": "usage",
            "provider_attribution": {
                "expected_item_count": 4,
                "attributed_item_count": 4,
                "percent": 100.0,
                "denominator_sha256": denominator,
            },
            "fallback_counts": {key: 0 for key in sorted(FALLBACK_KEYS)},
        },
    }
    return {
        "model-lock.json": model_lock,
        "validation.json": validation,
        "product.json": product,
    }


def write_synthetic_package(
    out_dir: Path,
    documents: dict[str, dict[str, Any]],
    *,
    status: str = "PASS",
    reason: tuple[str, str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, document in documents.items():
        write_json(out_dir / filename, document)
    manifest = {
        **synthetic_envelope("manifest"),
        "validator_version": VALIDATOR_VERSION,
        "final_status": status,
        "terminal_reason": None if reason is None else {"code": reason[0], "detail": reason[1]},
        "terminal_line": expected_terminal_line(status, "qwen38-27b-fp8", out_dir),
        "receipts": {
            filename: reference_for(out_dir / filename, filename)
            for filename in ["model-lock.json", "validation.json", "product.json"]
        },
    }
    write_json(out_dir / "manifest.json", manifest)


def expect_failure(label: str, operation: Any, pattern: str) -> None:
    try:
        operation()
    except GateError as exc:
        require(pattern in str(exc), f"{label} failed for unexpected reason: {exc}")
        return
    raise GateError(f"self-test {label} unexpectedly passed")


def run_self_test() -> None:
    validate_schema_files()
    with tempfile.TemporaryDirectory(prefix="ferrum-vnext-adoption-selftest-") as raw:
        root = Path(raw)

        def blocked_documents(path: Path, reason: tuple[str, str]) -> dict[str, dict[str, Any]]:
            documents = synthetic_pass_documents(path)
            for stage, (filename, section) in STAGE_SECTIONS.items():
                documents[filename]["milestones"][stage] = {
                    "status": "not_run",
                    "reason_code": reason[0],
                    "reason": reason[1],
                }
                documents[filename][section] = None
            documents["validation.json"]["binary_sha256"] = None
            documents["validation.json"]["references"] = []
            documents["product.json"]["binary_sha256"] = None
            documents["product.json"]["hardware"] = None
            documents["product.json"]["effective_config"] = None
            return documents

        passing = root / "pass"
        pass_docs = synthetic_pass_documents(passing)
        write_synthetic_package(passing, pass_docs)
        expected = expected_terminal_line("PASS", "qwen38-27b-fp8", passing)
        require(validate_package("qwen38-27b-fp8", passing) == expected, "synthetic PASS line mismatch")

        threshold = root / "bad-threshold"
        threshold_docs = synthetic_pass_documents(threshold)
        threshold_docs["validation.json"]["numeric_approval"]["cases"][0]["relative_l2"] = 0.051
        write_synthetic_package(threshold, threshold_docs)
        expect_failure(
            "numeric threshold",
            lambda: validate_package("qwen38-27b-fp8", threshold, write_log=False),
            "exceeds 0.05",
        )

        tampered = root / "tampered"
        tampered_docs = synthetic_pass_documents(tampered)
        write_synthetic_package(tampered, tampered_docs)
        tampered_docs["product.json"]["usability"]["p50_ttft_seconds"] = 61.0
        write_json(tampered / "product.json", tampered_docs["product.json"])
        expect_failure(
            "receipt digest",
            lambda: validate_package("qwen38-27b-fp8", tampered, write_log=False),
            "SHA256 mismatch",
        )

        revision = root / "bad-revision"
        revision_docs = synthetic_pass_documents(revision)
        revision_docs["model-lock.json"]["checkpoint"]["revision"] = "0" * 40
        write_synthetic_package(revision, revision_docs)
        expect_failure(
            "checkpoint revision",
            lambda: validate_package("qwen38-27b-fp8", revision, write_log=False),
            "checkpoint revision mismatch",
        )

        blocked = root / "blocked"
        reason = ("source.unavailable", "locked checkpoint is unavailable")
        blocked_docs = blocked_documents(blocked, reason)
        write_synthetic_package(blocked, blocked_docs, status="BLOCKED", reason=reason)
        blocked_line = expected_terminal_line("BLOCKED", "qwen38-27b-fp8", blocked)
        require(
            validate_package("qwen38-27b-fp8", blocked, write_log=False) == blocked_line,
            "synthetic BLOCKED line mismatch",
        )

        rejected = root / "rejected"
        rejected_docs = blocked_documents(rejected, reason)
        rejected_docs["model-lock.json"]["milestones"]["M0"]["status"] = "fail"
        write_synthetic_package(rejected, rejected_docs, status="REJECT", reason=reason)
        rejected_line = expected_terminal_line("REJECT", "qwen38-27b-fp8", rejected)
        require(
            validate_package("qwen38-27b-fp8", rejected, write_log=False) == rejected_line,
            "synthetic REJECT line mismatch",
        )

        split_reason = root / "split-reason"
        split_docs = blocked_documents(split_reason, reason)
        split_docs["product.json"]["milestones"]["M5"]["reason"] = "a second reason"
        write_synthetic_package(split_reason, split_docs, status="REJECT", reason=reason)
        expect_failure(
            "unique terminal reason",
            lambda: validate_package("qwen38-27b-fp8", split_reason, write_log=False),
            "unique terminal reason",
        )

        schema = root / "bad-schema"
        schema_docs = synthetic_pass_documents(schema)
        del schema_docs["product.json"]["timing"]
        write_synthetic_package(schema, schema_docs)
        expect_failure(
            "schema envelope",
            lambda: validate_package("qwen38-27b-fp8", schema, write_log=False),
            "missing required property 'timing'",
        )
    print("VNEXT MODEL ADOPTION GATE SELF-TEST PASS")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_id", nargs="?", choices=sorted(CHECKPOINTS))
    parser.add_argument("out_dir", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.checkpoint_id is not None or args.out_dir is not None:
            parser.error("--self-test cannot be combined with checkpoint_id/out_dir")
        run_self_test()
        return 0
    if args.checkpoint_id is None or args.out_dir is None:
        parser.error("checkpoint_id and out_dir are required unless --self-test is used")
    try:
        terminal_line = validate_package(args.checkpoint_id, args.out_dir)
    except GateError as exc:
        if args.out_dir.is_dir():
            (args.out_dir / "validator.log").write_text(
                f"validator_version={VALIDATOR_VERSION}\nvalidation_error={exc}\n",
                encoding="utf-8",
            )
        print(f"vNext model-adoption gate failed: {exc}", file=sys.stderr)
        return 1
    print(terminal_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
