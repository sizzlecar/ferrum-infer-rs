#!/usr/bin/env python3
"""Validate and freeze the S1-derived CUDA live catalog consumed by G07B.

This is deliberately narrower than the full G03 goal.  It proves that one
clean-source S1 CUDA artifact and one bounded live-catalog export agree on
source, hardware, command, and catalog identity.  The native provider catalog
is preserved byte-for-byte so G07B can bind its package receipts to the exact
canonical catalog SHA256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence

from runtime_vnext_native_operator_set import (
    PUBLIC_IDENTITY_FIELDS as NATIVE_OPERATOR_SET_PUBLIC_IDENTITY_FIELDS,
    NativeOperatorSetEvidenceError,
    create_selftest_native_operator_set,
    public_identity as native_operator_set_public_identity,
    validate_native_operator_set,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
ARTIFACT_TYPE = "runtime_vnext_g03_live_catalog_checkpoint"
RAW_ARTIFACT_TYPE = "runtime_vnext_g03_live_catalog_raw_collection"
LANE = "runtime-vnext-g03-live-catalog"
PASS_PREFIX = "FERRUM RUNTIME VNEXT G03 LIVE CATALOG PASS"
BOUNDED_RECEIPT_SCHEMA = "ferrum.bounded-command-receipt.v1"
READY_PREFIX = "FERRUM RUNTIME VNEXT CUDA NATIVE CATALOG INPUT READY:"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_LOG_BYTES = 16 * 1024 * 1024
MAX_CATALOG_ROWS = 16_384
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
IDENTITY_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,160}$")

PROVIDER_CATALOG_FIELDS = ("schema_version", "backend", "providers")
PROVIDER_ROW_FIELDS = (
    "operation_id",
    "operation_contract_version",
    "operation_fingerprint",
    "provider_id",
    "provider_version",
    "provider_implementation_fingerprint",
)
CAPABILITY_CATALOG_FIELDS = (
    "device",
    "operations",
    "providers",
    "engine_providers",
    "weight_materializers",
)
DEVICE_FIELDS = (
    "id",
    "class",
    "ordinal",
    "total_memory_bytes",
    "runtime_implementation_fingerprint",
    "capabilities",
    "dynamic_storage_profiles",
)
OPERATION_FIELDS = (
    "id",
    "version",
    "inputs",
    "outputs",
    "attributes",
    "resources",
    "oracle",
    "provider",
    "profile_phase",
)
TENSOR_FIELDS = ("dimensions", "element_types", "layouts", "access", "alias")
RESOURCE_FIELDS = (
    "minimum_value_alignment_bytes",
    "scratch",
    "binding",
    "persistent",
)
PROVIDER_REQUIREMENT_FIELDS = ("minimum_version", "required_capabilities")
OPERATION_PROVIDER_FIELDS = (
    "provider_id",
    "operation_id",
    "operation_fingerprint",
    "provider_implementation_fingerprint",
    "execution_semantics",
    "version",
    "device_id",
    "capabilities",
    "accepted_weight_formats",
    "accepted_quantization_formats",
    "dynamic_storage_bindings",
    "resource_estimator_id",
    "resource_estimator_version",
    "resource_estimator_implementation_fingerprint",
)
EXECUTION_SEMANTICS_FIELDS = (
    "contract_version",
    "contract_fingerprint",
    "repeatability",
    "replay_equivalence",
)
STORAGE_BINDING_FIELDS = ("role", "ordinal", "storage")
STORAGE_REQUIREMENT_FIELDS = ("accepted_profiles",)
DYNAMIC_PROFILE_FIELDS = ("allocator", "view")
ENGINE_PROVIDER_FIELDS = (
    "provider_id",
    "contract_version",
    "implementation_fingerprint",
    "device_id",
    "capabilities",
)
MATERIALIZER_FIELDS = (
    "id",
    "version",
    "implementation_fingerprint",
    "fidelity",
    "required_capabilities",
)
RAW_MANIFEST_FIELDS = {
    "schema_version",
    "artifact_type",
    "status",
    "created_at",
    "source",
    "hardware",
    "collector",
    "scope",
    "bootstrap_native_operator_set",
    "build",
    "export",
    "does_not_prove",
    "artifacts",
    "artifact_count",
}
RAW_SCOPE_FIELDS = {
    "backend",
    "gpu_count",
    "gpu_model",
    "cuda_ordinal",
    "attention_policy",
    "cargo_profile",
    "cargo_jobs",
    "features",
}
RAW_EXPORT_FIELDS = {
    "command",
    "portable_command",
    "binary",
    "receipt",
    "receipt_status",
    "readiness",
    "provider_catalog",
    "capability_catalog",
}
RAW_COLLECTOR_FIELDS = {"source_path", "path", "sha256", "size_bytes"}
RAW_BOOTSTRAP_FIELDS = {
    "role",
    "lock",
    *NATIVE_OPERATOR_SET_PUBLIC_IDENTITY_FIELDS,
}
RAW_BOOTSTRAP_CLOSURE_FIELDS = {"member_count", "total_bytes", "index_sha256"}
RAW_BUILD_FIELDS = {
    "command",
    "portable_command",
    "receipt",
    "summary",
    "summary_receipt",
    "native_build_cache",
    "native_import_dirs",
}
RAW_BUILD_SUMMARY_FIELDS = {
    "sha256",
    "size_bytes",
    "row_count",
    "native_operator_artifact_set_status",
    "native_operator_artifact_set_inputs_hash",
    "core_ptx_count",
    "core_ptx_status",
    "core_ptx_artifacts_sha256",
}
RAW_READINESS_FIELDS = {"line", "provider_count", "capability_fingerprint"}
SOURCE_FIELDS = {"git_sha", "git_tree_sha", "dirty", "status_short"}
HARDWARE_FIELDS = {
    "policy",
    "gpu_count",
    "gpu",
    "nvidia_smi",
    "nvcc",
    "cargo",
    "rustc",
    "tools",
}
FILE_IDENTITY_FIELDS = {"path", "sha256", "size_bytes"}
PLAN_FIELDS = {
    "schema_version",
    "step_id",
    "command",
    "cwd",
    "expected_duration_seconds",
    "hard_deadline_seconds",
    "progress_signal",
    "worker_limits",
    "started_at",
}
RECEIPT_FIELDS = {
    "schema",
    "command",
    "cwd",
    "pid",
    "pgid",
    "limits",
    "peaks",
    "started_at",
    "ended_at",
    "duration_seconds",
    "reason",
    "rc",
    "status",
    "successful_samples",
    "sampling_error_count",
    "sampling_errors",
    "violation",
    "termination",
    "cleanup",
    "stdout",
    "stderr",
}
LIMIT_FIELDS = {
    "wall_timeout_seconds",
    "max_processes",
    "max_group_threads",
    "max_per_process_threads",
    "sample_interval_seconds",
    "max_sampling_errors",
    "term_grace_seconds",
}
PEAK_FIELDS = {
    "processes",
    "group_threads",
    "per_process_threads",
    "per_process_threads_pid",
}
REQUIRED_CUDA_NATIVE_OPERATORS = (
    "ferrum.cuda.marlin",
    "ferrum.cuda.vllm_marlin",
    "ferrum.cuda.vllm_moe_marlin",
    "ferrum.cuda.vllm_paged_attention_v2",
)
REQUIRED_CUDA_NATIVE_BUILD_UNITS = (
    ("marlin", "marlin", "ferrum.cuda.marlin"),
    ("vllm_marlin", "vllm_marlin", "ferrum.cuda.vllm_marlin"),
    ("vllm_moe_marlin", "vllm_moe_marlin", "ferrum.cuda.vllm_moe_marlin"),
    (
        "vllm_paged_attn",
        "vllm_paged_attention_v2",
        "ferrum.cuda.vllm_paged_attention_v2",
    ),
)
CORE_PTX_BLOCK_RE = re.compile(
    r"const CORE_PTX_KERNELS:\s*&\[&str\]\s*=\s*&\[(?P<body>.*?)\];",
    re.DOTALL,
)
QUOTED_RUST_PATH_RE = re.compile(r'"([^"\r\n]+)"')
RAW_DOES_NOT_PROVE = [
    "canonical G03 PASS",
    "full G03 CPU/CUDA/Metal conformance",
    "canonical G07B PASS",
    "model correctness",
    "model performance",
    "release readiness",
]
DOES_NOT_PROVE = [
    "full G03",
    "conformance",
    "determinism",
    "performance",
    "G07B PASS",
    "release readiness",
]


class CheckpointError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def run_text(cwd: Path, command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CheckpointError(f"cannot run {list(command)!r}: {error}") from error
    require(
        result.returncode == 0,
        f"command failed ({result.returncode}): {list(command)!r}: {result.stderr[-1000:]}",
    )
    return result.stdout.strip()


def source_identity(source_root: Path) -> dict[str, Any]:
    require(
        source_root.is_dir() and not source_root.is_symlink(),
        f"source root is not a real directory: {source_root}",
    )
    git_sha = run_text(source_root, ["git", "rev-parse", "HEAD"])
    tree_sha = run_text(source_root, ["git", "rev-parse", "HEAD^{tree}"])
    status_short = run_text(
        source_root,
        ["git", "status", "--short", "--untracked-files=all"],
    ).splitlines()
    require(GIT_SHA_RE.fullmatch(git_sha) is not None, "invalid source Git SHA")
    require(GIT_SHA_RE.fullmatch(tree_sha) is not None, "invalid source Git tree SHA")
    require(not status_short, f"G03 live catalog requires clean source: {status_short}")
    return {
        "git_sha": git_sha,
        "git_tree_sha": tree_sha,
        "dirty": False,
        "status_short": [],
    }


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_float(raw: str) -> Any:
    raise CheckpointError(f"floating-point JSON number is not canonical: {raw}")


def _reject_constant(raw: str) -> Any:
    raise CheckpointError(f"non-finite JSON value is not allowed: {raw}")


def read_regular_bytes(path: Path, max_bytes: int, label: str) -> bytes:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise CheckpointError(f"cannot open {label} {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        require(
            stat.S_ISREG(metadata.st_mode) and metadata.st_size <= max_bytes,
            f"{label} is not regular or exceeds {max_bytes} bytes: {path}",
        )
        chunks: list[bytes] = []
        total = 0
        while total <= max_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        require(total <= max_bytes, f"{label} grew beyond {max_bytes} bytes: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def parse_json_bytes(raw: bytes, label: str, *, allow_floats: bool = False) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_float=float if allow_floats else _reject_float,
            parse_constant=_reject_constant,
        )
    except CheckpointError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot parse {label}: {error}") from error


def read_json(path: Path, label: str, max_bytes: int = MAX_JSON_BYTES) -> tuple[Any, bytes]:
    raw = read_regular_bytes(path, max_bytes, label)
    return parse_json_bytes(raw, label, allow_floats=True), raw


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"artifact is not regular: {path}")
    resolved = path.resolve()
    display = str(resolved)
    if relative_to is not None:
        root = relative_to.resolve()
        require(resolved.is_relative_to(root), f"artifact escapes root: {path}")
        display = resolved.relative_to(root).as_posix()
    return {
        "path": display,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256_bytes(encoded)


def identity_materializer_fingerprint() -> str:
    return canonical_json_sha256(
        {
            "id": "weight-materializer.identity",
            "version": {"major": 2, "minor": 0},
            "contract": "execution-weight-plan.identity.v2",
        }
    )


def rust_compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def rust_pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def require_dict(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def require_exact_fields(value: dict[str, Any], fields: set[str], label: str) -> None:
    require(set(value) == fields, f"{label} field set mismatch: {sorted(set(value) ^ fields)}")


def require_ordered_fields(
    value: dict[str, Any], fields: tuple[str, ...], label: str
) -> None:
    require(
        tuple(value) == fields,
        f"{label} field order/shape mismatch: {tuple(value)!r}",
    )


def require_sha(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA256",
    )
    return value


def require_identity(value: Any, prefix: str, label: str) -> str:
    require(
        isinstance(value, str)
        and value.startswith(prefix)
        and IDENTITY_RE.fullmatch(value) is not None,
        f"{label} is invalid",
    )
    return value


def require_int(value: Any, label: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} is invalid",
    )
    if maximum is not None:
        require(value <= maximum, f"{label} exceeds {maximum}")
    return value


def validate_version(value: Any, label: str, *, positive_major: bool = True) -> tuple[int, int]:
    row = require_dict(value, label)
    require_ordered_fields(row, ("major", "minor"), label)
    major = require_int(row["major"], f"{label}.major", minimum=1 if positive_major else 0, maximum=65535)
    minor = require_int(row["minor"], f"{label}.minor", maximum=65535)
    return major, minor


def version_satisfies(available: tuple[int, int], required: tuple[int, int]) -> bool:
    return available[0] == required[0] and available[1] >= required[1]


def validate_sorted_strings(
    value: Any,
    label: str,
    *,
    nonempty: bool = False,
    prefix: str | None = None,
) -> list[str]:
    rows = require_list(value, label)
    require(not nonempty or rows, f"{label} must not be empty")
    require(
        all(isinstance(row, str) and row and IDENTITY_RE.fullmatch(row) for row in rows),
        f"{label} contains an invalid identity",
    )
    if prefix is not None:
        require(all(row.startswith(prefix) for row in rows), f"{label} prefix mismatch")
    require(rows == sorted(set(rows)), f"{label} must be sorted and unique")
    return rows


def validate_element_types(value: Any, label: str) -> list[str]:
    rows = require_list(value, label)
    order = {
        name: index
        for index, name in enumerate(
            ("bool", "u8", "u32", "i8", "i32", "f16", "bf16", "f32")
        )
    }
    require(rows and all(row in order for row in rows), f"{label} is invalid")
    require(
        rows == sorted(set(rows), key=order.__getitem__),
        f"{label} must follow ElementType's canonical Rust order",
    )
    return rows


def validate_dynamic_profile(value: Any, label: str) -> None:
    row = require_dict(value, label)
    require_ordered_fields(row, DYNAMIC_PROFILE_FIELDS, label)
    allocator = row["allocator"]
    view = row["view"]
    require(
        allocator == "linear_arena"
        or (isinstance(allocator, dict) and set(allocator) == {"fixed_block_arena"}),
        f"{label}.allocator is invalid",
    )
    require(
        view == "contiguous"
        or (isinstance(view, dict) and set(view) == {"paged_regions"}),
        f"{label}.view is invalid",
    )


def validate_rational(value: Any, label: str) -> None:
    row = require_dict(value, label)
    require_ordered_fields(row, ("numerator", "denominator"), label)
    numerator = row["numerator"]
    require(isinstance(numerator, int) and not isinstance(numerator, bool), f"{label}.numerator is invalid")
    denominator = require_int(row["denominator"], f"{label}.denominator", minimum=1)
    require(math.gcd(abs(numerator), denominator) == 1, f"{label} is not reduced")


def validate_dimension(value: Any, label: str) -> None:
    row = require_dict(value, label)
    require(len(row) == 1, f"{label} must contain one dimension variant")
    kind, payload = next(iter(row.items()))
    if kind == "exact":
        require_int(payload, f"{label}.exact")
    elif kind == "symbol":
        require(isinstance(payload, str) and IDENTITY_RE.fullmatch(payload) is not None, f"{label}.symbol is invalid")
    elif kind == "range":
        bounds = require_dict(payload, f"{label}.range")
        require_ordered_fields(bounds, ("minimum", "maximum"), f"{label}.range")
        minimum = require_int(bounds["minimum"], f"{label}.range.minimum")
        maximum = require_int(bounds["maximum"], f"{label}.range.maximum")
        require(minimum <= maximum, f"{label}.range is reversed")
    else:
        raise CheckpointError(f"{label} has an unknown dimension variant: {kind}")


def validate_stride(value: Any, label: str) -> tuple[int, Any]:
    row = require_dict(value, label)
    require(len(row) == 1, f"{label} must contain one stride variant")
    kind, payload = next(iter(row.items()))
    if kind == "exact_bytes":
        return (0, require_int(payload, f"{label}.exact_bytes"))
    if kind == "symbol":
        require(isinstance(payload, str) and IDENTITY_RE.fullmatch(payload) is not None, f"{label}.symbol is invalid")
        return (1, payload)
    raise CheckpointError(f"{label} has an unknown stride variant: {kind}")


def validate_layout(value: Any, label: str) -> tuple[Any, ...]:
    if value == "contiguous":
        return (0,)
    row = require_dict(value, label)
    require(len(row) == 1, f"{label} must contain one layout variant")
    kind, payload = next(iter(row.items()))
    fields = require_dict(payload, f"{label}.{kind}")
    if kind == "strided":
        require_ordered_fields(fields, ("strides",), f"{label}.strided")
        strides = require_list(fields["strides"], f"{label}.strided.strides")
        require(strides, f"{label}.strided.strides is empty")
        return (1, tuple(validate_stride(item, f"{label}.strides[{index}]") for index, item in enumerate(strides)))
    if kind == "blocked":
        require_ordered_fields(fields, ("block", "axis_order"), f"{label}.blocked")
        block = tuple(require_int(item, f"{label}.blocked.block[{index}]", minimum=1) for index, item in enumerate(require_list(fields["block"], f"{label}.blocked.block")))
        axes = tuple(require_int(item, f"{label}.blocked.axis_order[{index}]") for index, item in enumerate(require_list(fields["axis_order"], f"{label}.blocked.axis_order")))
        require(block and len(block) == len(axes) and set(axes) == set(range(len(axes))), f"{label}.blocked shape/axis order is invalid")
        return (2, block, axes)
    raise CheckpointError(f"{label} has an unknown layout variant: {kind}")


def validate_attribute_constraint(value: Any, label: str) -> None:
    if value == "none" or (isinstance(value, dict) and set(value) == {"bool_equals"} and isinstance(value["bool_equals"], bool)):
        return
    row = require_dict(value, label)
    require(len(row) == 1, f"{label} must contain one constraint variant")
    kind, payload = next(iter(row.items()))
    fields = require_dict(payload, f"{label}.{kind}")
    if kind in {"integer_range", "unsigned_range", "integer_list_length"}:
        require_ordered_fields(fields, ("minimum", "maximum"), f"{label}.{kind}")
        minimum = fields["minimum"]
        maximum = fields["maximum"]
        require(isinstance(minimum, int) and not isinstance(minimum, bool), f"{label}.{kind}.minimum is invalid")
        require(isinstance(maximum, int) and not isinstance(maximum, bool) and minimum <= maximum, f"{label}.{kind}.maximum is invalid")
        if kind != "integer_range":
            require(minimum >= 0, f"{label}.{kind}.minimum is negative")
        return
    if kind == "rational_range":
        require_ordered_fields(fields, ("minimum", "maximum"), f"{label}.{kind}")
        validate_rational(fields["minimum"], f"{label}.{kind}.minimum")
        validate_rational(fields["maximum"], f"{label}.{kind}.maximum")
        return
    if kind == "text_choices":
        require_ordered_fields(fields, ("values",), f"{label}.{kind}")
        values = require_list(fields["values"], f"{label}.{kind}.values")
        require(values and all(isinstance(item, str) for item in values) and values == sorted(set(values)), f"{label}.{kind}.values is invalid")
        return
    raise CheckpointError(f"{label} has an unknown attribute constraint: {kind}")


def validate_attributes(value: Any, label: str) -> None:
    attributes = require_dict(value, label)
    require_ordered_fields(attributes, ("entries",), label)
    entries = require_dict(attributes["entries"], f"{label}.entries")
    require(list(entries) == sorted(entries), f"{label}.entries is not sorted")
    for attribute_id, raw_spec in entries.items():
        require(isinstance(attribute_id, str) and IDENTITY_RE.fullmatch(attribute_id) is not None, f"{label} attribute ID is invalid")
        spec = require_dict(raw_spec, f"{label}.entries[{attribute_id}]")
        require_ordered_fields(spec, ("value_kind", "required", "constraint"), f"{label}.entries[{attribute_id}]")
        require(spec["value_kind"] in {"bool", "integer", "unsigned", "rational", "text", "integers"}, f"{label}.entries[{attribute_id}].value_kind is invalid")
        require(isinstance(spec["required"], bool), f"{label}.entries[{attribute_id}].required is invalid")
        validate_attribute_constraint(spec["constraint"], f"{label}.entries[{attribute_id}].constraint")


def validate_tensor(
    value: Any, label: str, *, output: bool, input_count: int = 0
) -> None:
    row = require_dict(value, label)
    require_ordered_fields(row, TENSOR_FIELDS, label)
    dimensions = require_list(row["dimensions"], f"{label}.dimensions")
    for index, dimension in enumerate(dimensions):
        validate_dimension(dimension, f"{label}.dimensions[{index}]")
    validate_element_types(row["element_types"], f"{label}.element_types")
    layouts = require_list(row["layouts"], f"{label}.layouts")
    require(layouts, f"{label}.layouts must not be empty")
    layout_keys = [
        validate_layout(item, f"{label}.layouts[{index}]")
        for index, item in enumerate(layouts)
    ]
    require(
        layout_keys == sorted(set(layout_keys)),
        f"{label}.layouts are not canonical and unique",
    )
    access = row["access"]
    require(access in {"read", "write", "read_write"}, f"{label}.access is invalid")
    require(
        access in ({"write", "read_write"} if output else {"read", "read_write"}),
        f"{label}.access is incompatible with its role",
    )
    alias = row["alias"]
    valid_alias = alias == "no_alias" or (
        isinstance(alias, dict)
        and len(alias) == 1
        and next(iter(alias)) in {"may_alias", "must_alias"}
        and isinstance(next(iter(alias.values())), dict)
        and set(next(iter(alias.values()))) == {"tensor_index"}
    )
    require(valid_alias, f"{label}.alias is invalid")
    if not output:
        require(alias == "no_alias", f"{label} input aliases another tensor")
    elif isinstance(alias, dict):
        tensor_index = next(iter(alias.values()))["tensor_index"]
        require_int(tensor_index, f"{label}.alias.tensor_index")
        require(tensor_index < input_count, f"{label}.alias.tensor_index is out of range")


def validate_operation(value: Any, operation_id: str, label: str) -> dict[str, Any]:
    row = require_dict(value, label)
    require_ordered_fields(row, OPERATION_FIELDS, label)
    require(row["id"] == operation_id, f"{label}.id differs from its map key")
    version = validate_version(row["version"], f"{label}.version")
    inputs = require_list(row["inputs"], f"{label}.inputs")
    outputs = require_list(row["outputs"], f"{label}.outputs")
    require(outputs, f"{label}.outputs must not be empty")
    for index, tensor in enumerate(inputs):
        validate_tensor(tensor, f"{label}.inputs[{index}]", output=False)
    for index, tensor in enumerate(outputs):
        validate_tensor(
            tensor,
            f"{label}.outputs[{index}]",
            output=True,
            input_count=len(inputs),
        )
    validate_attributes(row["attributes"], f"{label}.attributes")
    resources = require_dict(row["resources"], f"{label}.resources")
    require_ordered_fields(resources, RESOURCE_FIELDS, f"{label}.resources")
    alignment = require_int(
        resources["minimum_value_alignment_bytes"],
        f"{label}.resources.minimum_value_alignment_bytes",
        minimum=1,
    )
    require(alignment & (alignment - 1) == 0, f"{label} alignment is not a power of two")
    for field in ("scratch", "binding", "persistent"):
        require(
            resources[field] in {"forbidden", "optional", "required"},
            f"{label}.resources.{field} is invalid",
        )
    oracle = row["oracle"]
    require(
        oracle == "exact"
        or (
            isinstance(oracle, dict)
            and len(oracle) == 1
            and next(iter(oracle))
            in {"absolute_tolerance", "relative_tolerance", "reference_operation"}
        ),
        f"{label}.oracle is invalid",
    )
    provider = require_dict(row["provider"], f"{label}.provider")
    require_ordered_fields(provider, PROVIDER_REQUIREMENT_FIELDS, f"{label}.provider")
    minimum_version = validate_version(
        provider["minimum_version"], f"{label}.provider.minimum_version"
    )
    require(
        minimum_version[0] == version[0],
        f"{label} provider minimum major differs from operation major",
    )
    required_capabilities = validate_sorted_strings(
        provider["required_capabilities"],
        f"{label}.provider.required_capabilities",
    )
    require(
        row["profile_phase"]
        in {"load", "prepare", "forward", "prefill", "decode", "transfer", "synchronize"},
        f"{label}.profile_phase is invalid",
    )
    return {
        "version": version,
        "minimum_provider_version": minimum_version,
        "required_capabilities": set(required_capabilities),
        "input_count": len(inputs),
        "output_count": len(outputs),
        "fingerprint": sha256_bytes(rust_compact_json_bytes(row)),
        "oracle": oracle,
    }


def execution_semantics_fingerprint(
    version: tuple[int, int], repeatability: str, replay_equivalence: str
) -> str:
    repeatability_byte = {"bitwise_same_runtime": 1}.get(repeatability)
    replay_byte = {"ineligible": 0, "bitwise_eager_equivalent": 1}.get(
        replay_equivalence
    )
    require(repeatability_byte is not None, "unknown provider repeatability")
    require(replay_byte is not None, "unknown provider replay equivalence")
    digest = hashlib.sha256()
    digest.update(b"ferrum.runtime-vnext.provider-execution-semantics.v1\0")
    digest.update(struct.pack("<H", version[0]))
    digest.update(struct.pack("<H", version[1]))
    digest.update(bytes([repeatability_byte, replay_byte]))
    return digest.hexdigest()


def validate_operation_provider(
    value: Any,
    *,
    operation_id: str,
    operation: dict[str, Any],
    device_id: str,
    device_capabilities: set[str],
    label: str,
) -> dict[str, Any]:
    row = require_dict(value, label)
    require_ordered_fields(row, OPERATION_PROVIDER_FIELDS, label)
    provider_id = require_identity(row["provider_id"], "provider.cuda.", f"{label}.provider_id")
    require(row["operation_id"] == operation_id, f"{label}.operation_id mismatch")
    require(
        row["operation_fingerprint"] == operation["fingerprint"],
        f"{label}.operation_fingerprint is stale or forged",
    )
    require_sha(
        row["provider_implementation_fingerprint"],
        f"{label}.provider_implementation_fingerprint",
    )
    semantics = require_dict(row["execution_semantics"], f"{label}.execution_semantics")
    require_ordered_fields(
        semantics, EXECUTION_SEMANTICS_FIELDS, f"{label}.execution_semantics"
    )
    semantics_version = validate_version(
        semantics["contract_version"], f"{label}.execution_semantics.contract_version"
    )
    require(semantics_version == (1, 0), f"{label} execution semantics version is unsupported")
    expected_semantics = execution_semantics_fingerprint(
        semantics_version, semantics["repeatability"], semantics["replay_equivalence"]
    )
    require(
        semantics["contract_fingerprint"] == expected_semantics,
        f"{label} execution semantics fingerprint mismatch",
    )
    version = validate_version(row["version"], f"{label}.version")
    require(
        version_satisfies(version, operation["version"])
        and version_satisfies(version, operation["minimum_provider_version"]),
        f"{label}.version does not satisfy its operation",
    )
    require(row["device_id"] == device_id, f"{label}.device_id mismatch")
    capabilities = set(
        validate_sorted_strings(row["capabilities"], f"{label}.capabilities")
    )
    require(capabilities <= device_capabilities, f"{label} advertises absent device capabilities")
    require(
        operation["required_capabilities"] <= capabilities,
        f"{label} lacks an operation-required capability",
    )
    validate_sorted_strings(row["accepted_weight_formats"], f"{label}.accepted_weight_formats")
    validate_sorted_strings(
        row["accepted_quantization_formats"],
        f"{label}.accepted_quantization_formats",
    )
    bindings = require_list(row["dynamic_storage_bindings"], f"{label}.dynamic_storage_bindings")
    require(bindings, f"{label}.dynamic_storage_bindings must not be empty")
    binding_keys: list[tuple[int, int]] = []
    for index, raw_binding in enumerate(bindings):
        binding_label = f"{label}.dynamic_storage_bindings[{index}]"
        binding = require_dict(raw_binding, binding_label)
        require_ordered_fields(binding, STORAGE_BINDING_FIELDS, binding_label)
        role = binding["role"]
        require(role in {"input", "output"}, f"{binding_label}.role is invalid")
        ordinal = require_int(binding["ordinal"], f"{binding_label}.ordinal")
        limit = operation["input_count"] if role == "input" else operation["output_count"]
        require(ordinal < limit, f"{binding_label}.ordinal is out of range")
        storage = require_dict(binding["storage"], f"{binding_label}.storage")
        require_ordered_fields(storage, STORAGE_REQUIREMENT_FIELDS, f"{binding_label}.storage")
        profiles = require_list(storage["accepted_profiles"], f"{binding_label}.storage.accepted_profiles")
        require(profiles, f"{binding_label} accepts no storage profile")
        profile_keys: list[bytes] = []
        for profile_index, profile in enumerate(profiles):
            validate_dynamic_profile(profile, f"{binding_label}.profiles[{profile_index}]")
            profile_keys.append(rust_compact_json_bytes(profile))
        require(len(profile_keys) == len(set(profile_keys)), f"{binding_label} profiles duplicate")
        binding_keys.append((0 if role == "input" else 1, ordinal))
    require(binding_keys == sorted(set(binding_keys)), f"{label} storage bindings are not canonical")
    estimator_id = row["resource_estimator_id"]
    require(
        isinstance(estimator_id, str) and IDENTITY_RE.fullmatch(estimator_id) is not None,
        f"{label}.resource_estimator_id is invalid",
    )
    validate_version(row["resource_estimator_version"], f"{label}.resource_estimator_version")
    require_sha(
        row["resource_estimator_implementation_fingerprint"],
        f"{label}.resource_estimator_implementation_fingerprint",
    )
    return {
        "operation_id": operation_id,
        "operation_contract_version": {
            "major": operation["version"][0],
            "minor": operation["version"][1],
        },
        "operation_fingerprint": operation["fingerprint"],
        "provider_id": provider_id,
        "provider_version": {"major": version[0], "minor": version[1]},
        "provider_implementation_fingerprint": row[
            "provider_implementation_fingerprint"
        ],
    }


def validate_capability_catalog_bytes(raw: bytes, *, cuda_ordinal: int) -> dict[str, Any]:
    value = require_dict(parse_json_bytes(raw, "capability catalog"), "capability catalog")
    require_ordered_fields(value, CAPABILITY_CATALOG_FIELDS, "capability catalog")
    require(raw == rust_pretty_json_bytes(value), "capability catalog is not canonical pretty JSON + LF")
    device = require_dict(value["device"], "capability catalog.device")
    require_ordered_fields(device, DEVICE_FIELDS, "capability catalog.device")
    require(device["id"] == f"cuda:{cuda_ordinal}", "capability device ID mismatch")
    require(device["class"] == "accelerator", "capability device class mismatch")
    require(device["ordinal"] == cuda_ordinal, "capability device ordinal mismatch")
    require_int(device["total_memory_bytes"], "capability device memory", minimum=1)
    require_sha(device["runtime_implementation_fingerprint"], "device runtime fingerprint")
    device_capabilities = set(
        validate_sorted_strings(
            device["capabilities"], "device capabilities", nonempty=True
        )
    )
    profiles = require_list(device["dynamic_storage_profiles"], "device dynamic profiles")
    require(profiles, "device dynamic storage profiles are empty")
    profile_keys: list[bytes] = []
    for index, profile in enumerate(profiles):
        validate_dynamic_profile(profile, f"device dynamic profiles[{index}]")
        profile_keys.append(rust_compact_json_bytes(profile))
    require(len(profile_keys) == len(set(profile_keys)), "device dynamic profiles duplicate")

    operations = require_dict(value["operations"], "capability catalog.operations")
    providers = require_dict(value["providers"], "capability catalog.providers")
    engines = require_dict(value["engine_providers"], "capability catalog.engine_providers")
    materializers = require_dict(
        value["weight_materializers"], "capability catalog.weight_materializers"
    )
    for name, rows in (
        ("operations", operations),
        ("providers", providers),
        ("engine_providers", engines),
        ("weight_materializers", materializers),
    ):
        require(rows, f"capability catalog.{name} is empty")
        require(list(rows) == sorted(rows), f"capability catalog.{name} is not sorted")
    require(set(operations) == set(providers), "operation/provider map denominators differ")
    require(len(operations) <= 4096, "operation catalog exceeds row budget")

    operation_summaries: dict[str, dict[str, Any]] = {}
    for operation_id, operation in operations.items():
        require_identity(operation_id, "operation.", f"operation key {operation_id!r}")
        operation_summaries[operation_id] = validate_operation(
            operation, operation_id, f"operations[{operation_id}]"
        )

    projection: list[dict[str, Any]] = []
    provider_count = 0
    for operation_id, rows_value in providers.items():
        rows = require_list(rows_value, f"providers[{operation_id}]")
        require(rows, f"providers[{operation_id}] is empty")
        provider_count += len(rows)
        require(provider_count <= MAX_CATALOG_ROWS, "provider catalog exceeds row budget")
        provider_keys: list[tuple[str, tuple[int, int]]] = []
        for index, provider in enumerate(rows):
            summary = validate_operation_provider(
                provider,
                operation_id=operation_id,
                operation=operation_summaries[operation_id],
                device_id=device["id"],
                device_capabilities=device_capabilities,
                label=f"providers[{operation_id}][{index}]",
            )
            projection.append(summary)
            version = summary["provider_version"]
            provider_keys.append(
                (summary["provider_id"], (version["major"], version["minor"]))
            )
        require(provider_keys == sorted(provider_keys), f"providers[{operation_id}] is not sorted")
        require(
            len({key[0] for key in provider_keys}) == len(provider_keys),
            f"providers[{operation_id}] contains duplicate provider IDs",
        )

    for provider_id, raw_engine in engines.items():
        require_identity(provider_id, "provider.", f"engine provider key {provider_id!r}")
        engine = require_dict(raw_engine, f"engine_providers[{provider_id}]")
        require_ordered_fields(engine, ENGINE_PROVIDER_FIELDS, f"engine_providers[{provider_id}]")
        require(engine["provider_id"] == provider_id, f"engine provider {provider_id} key mismatch")
        validate_version(engine["contract_version"], f"engine provider {provider_id} version")
        require_sha(engine["implementation_fingerprint"], f"engine provider {provider_id} fingerprint")
        require(engine["device_id"] == device["id"], f"engine provider {provider_id} device mismatch")
        engine_caps = set(
            validate_sorted_strings(engine["capabilities"], f"engine provider {provider_id} capabilities")
        )
        require(engine_caps <= device_capabilities, f"engine provider {provider_id} capabilities mismatch")

    for materializer_id, raw_materializer in materializers.items():
        require_identity(
            materializer_id,
            "weight-materializer.",
            f"materializer key {materializer_id!r}",
        )
        materializer = require_dict(raw_materializer, f"materializers[{materializer_id}]")
        require_ordered_fields(materializer, MATERIALIZER_FIELDS, f"materializers[{materializer_id}]")
        require(materializer["id"] == materializer_id, f"materializer {materializer_id} key mismatch")
        validate_version(materializer["version"], f"materializer {materializer_id} version")
        require_sha(materializer["implementation_fingerprint"], f"materializer {materializer_id} fingerprint")
        require(
            materializer["fidelity"] in {"exact", "approximate"},
            f"materializer {materializer_id} fidelity invalid",
        )
        required = set(
            validate_sorted_strings(
                materializer["required_capabilities"],
                f"materializer {materializer_id} capabilities",
            )
        )
        require(required <= device_capabilities, f"materializer {materializer_id} capabilities mismatch")
    identity = materializers.get("weight-materializer.identity")
    require(isinstance(identity, dict), "canonical identity weight materializer is missing")
    require(
        identity["version"] == {"major": 2, "minor": 0}
        and identity["implementation_fingerprint"]
        == identity_materializer_fingerprint()
        and identity["fidelity"] == "exact"
        and identity["required_capabilities"] == [],
        "canonical identity weight materializer drifted",
    )

    projection.sort(key=lambda row: (row["operation_id"], row["provider_id"]))
    return {
        "value": value,
        "file_sha256": sha256_bytes(raw),
        "size_bytes": len(raw),
        "collector_fingerprint": canonical_json_sha256(value),
        "runtime_fingerprint": sha256_bytes(rust_compact_json_bytes(value)),
        "device_id": device["id"],
        "runtime_implementation_fingerprint": device[
            "runtime_implementation_fingerprint"
        ],
        "operations_count": len(operations),
        "providers_count": len(providers),
        "provider_row_count": provider_count,
        "engine_providers_count": len(engines),
        "weight_materializers_count": len(materializers),
        "operation_ids": list(operations),
        "projection": projection,
    }


def validate_provider_catalog_bytes(raw: bytes) -> dict[str, Any]:
    value = require_dict(parse_json_bytes(raw, "provider catalog"), "provider catalog")
    require_ordered_fields(value, PROVIDER_CATALOG_FIELDS, "provider catalog")
    require(value["schema_version"] == 1, "provider catalog schema must be 1")
    require(value["backend"] == "cuda", "provider catalog backend must be CUDA")
    providers = require_list(value["providers"], "provider catalog.providers")
    require(providers, "provider catalog is empty")
    require(len(providers) <= MAX_CATALOG_ROWS, "provider catalog exceeds row budget")
    keys: list[tuple[str, str]] = []
    canonical_rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(providers):
        label = f"provider catalog.providers[{index}]"
        row = require_dict(raw_row, label)
        require_ordered_fields(row, PROVIDER_ROW_FIELDS, label)
        operation_id = require_identity(row["operation_id"], "operation.", f"{label}.operation_id")
        provider_id = require_identity(row["provider_id"], "provider.cuda.", f"{label}.provider_id")
        validate_version(row["operation_contract_version"], f"{label}.operation_contract_version")
        validate_version(row["provider_version"], f"{label}.provider_version")
        require_sha(row["operation_fingerprint"], f"{label}.operation_fingerprint")
        require_sha(
            row["provider_implementation_fingerprint"],
            f"{label}.provider_implementation_fingerprint",
        )
        keys.append((operation_id, provider_id))
        canonical_rows.append({field: row[field] for field in PROVIDER_ROW_FIELDS})
    require(keys == sorted(set(keys)), "provider rows are not sorted and unique")
    canonical_value = {
        "schema_version": 1,
        "backend": "cuda",
        "providers": canonical_rows,
    }
    require(
        raw == rust_pretty_json_bytes(canonical_value),
        "provider catalog bytes are not Rust canonical pretty JSON + LF",
    )
    return {
        "value": value,
        "file_sha256": sha256_bytes(raw),
        "size_bytes": len(raw),
        "collector_fingerprint": canonical_json_sha256(value),
        "provider_count": len(providers),
        "operation_count": len({row["operation_id"] for row in providers}),
        "provider_ids": [row["provider_id"] for row in providers],
        "operation_ids": sorted({row["operation_id"] for row in providers}),
    }


def validate_file_identity(path: Path, value: Any, label: str) -> dict[str, Any]:
    row = require_dict(value, label)
    require_exact_fields(row, FILE_IDENTITY_FIELDS, label)
    require_sha(row["sha256"], f"{label}.sha256")
    require_int(row["size_bytes"], f"{label}.size_bytes")
    require(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == row["size_bytes"]
        and sha256_file(path) == row["sha256"],
        f"{label} file identity mismatch: {path}",
    )
    return row


def safe_relative_path(raw: Any, label: str) -> PurePosixPath:
    require(isinstance(raw, str) and raw, f"{label} must be a non-empty path")
    require("\\" not in raw, f"{label} must use portable separators")
    path = PurePosixPath(raw)
    require(
        not path.is_absolute()
        and path.as_posix() == raw
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{label} is not a safe relative path: {raw!r}",
    )
    return path


def resolve_artifact(root: Path, raw: Any, label: str) -> Path:
    relative = safe_relative_path(raw, f"{label}.path")
    candidate = root.joinpath(*relative.parts)
    require(
        candidate.is_file() and not candidate.is_symlink(),
        f"{label} is missing or symlinked: {candidate}",
    )
    require(candidate.resolve().is_relative_to(root.resolve()), f"{label} escapes artifact root")
    return candidate


def validate_relative_ref(root: Path, value: Any, label: str) -> Path:
    row = require_dict(value, label)
    require_exact_fields(row, FILE_IDENTITY_FIELDS, label)
    path = resolve_artifact(root, row["path"], label)
    validate_file_identity(path, row, label)
    return path


def external_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    identity = file_identity(resolved)
    identity["path"] = str(resolved)
    return identity


def validate_external_ref(value: Any, label: str) -> Path:
    row = require_dict(value, label)
    require_exact_fields(row, FILE_IDENTITY_FIELDS, label)
    raw_path = row["path"]
    require(isinstance(raw_path, str) and Path(raw_path).is_absolute(), f"{label}.path must be absolute")
    path = Path(raw_path)
    validate_file_identity(path, row, label)
    return path.resolve()


def validate_s1_manifest(path: Path, source: dict[str, Any]) -> dict[str, Any]:
    manifest, manifest_raw = read_json(path, "S1 manifest")
    value = require_dict(manifest, "S1 manifest")
    artifact_root = Path(str(value.get("artifact_dir", ""))).expanduser().resolve()
    require(
        value.get("schema_version") == 1
        and value.get("artifact_type") == "runtime_vnext_s1_cuda_basic_slice_manifest"
        and value.get("checkpoint_id") == "S1-CUDA-basic"
        and value.get("lane") == "runtime-vnext-s1-cuda"
        and value.get("status") == "pass"
        and value.get("backend") == "cuda"
        and value.get("entrypoints") == ["ferrum run", "ferrum serve"]
        and value.get("source_git_sha") == source["git_sha"]
        and path.resolve() == artifact_root / "manifest.json"
        and value.get("pass_line")
        == f"FERRUM RUNTIME VNEXT S1 CUDA BASIC SLICE PASS: {artifact_root}",
        "S1 manifest identity/status/source/PASS mismatch",
    )
    binary_sha = require_sha(value.get("binary_sha256"), "S1 binary SHA256")
    hardware = value.get("hardware")
    require(isinstance(hardware, str) and hardware.strip(), "S1 hardware identity is missing")
    metrics = require_dict(value.get("metrics"), "S1 metrics")
    gpu_uuid = metrics.get("gpu_uuid")
    require(
        isinstance(gpu_uuid, str) and gpu_uuid.startswith("GPU-"),
        "S1 GPU UUID is invalid",
    )
    validation_ref = require_dict(value.get("validation"), "S1 validation ref")
    validation_path = validate_relative_ref(artifact_root, validation_ref, "S1 validation ref")
    validation, _ = read_json(validation_path, "S1 validation")
    validation_value = require_dict(validation, "S1 validation")
    require(
        validation_value.get("schema_version") == 1
        and validation_value.get("artifact_type")
        == "runtime_vnext_s1_cuda_basic_slice_validation"
        and validation_value.get("status") == "pass"
        and validation_value.get("source_git_sha") == source["git_sha"]
        and validation_value.get("binary_sha256") == binary_sha
        and validation_value.get("hardware") == hardware,
        "S1 validation does not bind the manifest source/binary/hardware",
    )
    return {
        "manifest": {
            "path": str(path.resolve()),
            "sha256": sha256_bytes(manifest_raw),
            "size_bytes": len(manifest_raw),
        },
        "binary_sha256": binary_sha,
        "hardware": hardware,
        "gpu_uuid": gpu_uuid,
        "validation": external_ref(validation_path),
    }


def validate_s1_outer_manifest(path: Path, source: dict[str, Any]) -> dict[str, Any]:
    outer, outer_raw = read_json(path, "S1 outer gate manifest")
    value = require_dict(outer, "S1 outer gate manifest")
    root = path.parent.resolve()
    expected_child_pass = f"FERRUM RUNTIME VNEXT S1 CUDA BASIC SLICE PASS: {root}"
    require(
        path.name == "gate.manifest.json"
        and value.get("schema_version") == 1
        and value.get("lane") == "vnext-s1-cuda"
        and value.get("status") == "pass"
        and value.get("child_returncode") == 0
        and value.get("child_pass_line") == expected_child_pass
        and value.get("git_sha") == source["git_sha"]
        and value.get("dirty_status") == {"is_dirty": False, "status_short": []}
        and Path(str(value.get("artifact_dir", ""))).resolve() == root
        and value.get("pass_line") == f"FERRUM GATE vnext-s1-cuda PASS: {root}"
        and value.get("error") is None,
        "S1 outer gate lane/status/PASS/source identity mismatch",
    )
    child_artifacts = require_dict(
        value.get("child_artifacts"), "S1 outer child artifacts"
    )
    require(
        set(child_artifacts) == {"kind", "child_manifest"}
        and child_artifacts["kind"] == "delegated-manifest",
        "S1 outer child artifact shape mismatch",
    )
    child_ref = require_dict(
        child_artifacts["child_manifest"], "S1 outer child manifest ref"
    )
    require(
        set(child_ref) == {"path", "sha256"},
        "S1 outer child manifest ref field set mismatch",
    )
    child_path = Path(str(child_ref.get("path", ""))).expanduser().resolve()
    require(
        child_path == root / "manifest.json"
        and child_path.is_file()
        and not child_path.is_symlink()
        and require_sha(child_ref.get("sha256"), "S1 child manifest SHA256")
        == sha256_file(child_path),
        "S1 outer child manifest path/SHA mismatch",
    )
    child = validate_s1_manifest(child_path, source)
    require(
        read_json(child_path, "S1 child manifest")[0].get("pass_line")
        == expected_child_pass,
        "S1 child PASS differs from outer child PASS",
    )
    return {
        "outer_manifest": {
            "path": str(path.resolve()),
            "sha256": sha256_bytes(outer_raw),
            "size_bytes": len(outer_raw),
        },
        **child,
    }


def validate_hardware(value: Any, s1: dict[str, Any]) -> dict[str, Any]:
    hardware = require_dict(value, "raw hardware")
    require_exact_fields(hardware, HARDWARE_FIELDS, "raw hardware")
    require(
        hardware["policy"] == "cuda-g0-1x-rtx4090"
        and hardware["gpu_count"] == 1,
        "raw hardware policy/count mismatch",
    )
    gpu = hardware["gpu"]
    require(isinstance(gpu, str), "raw hardware GPU row is invalid")
    fields = [field.strip() for field in gpu.split(",")]
    require(
        len(fields) == 5
        and fields[0] == "0"
        and "RTX 4090" in fields[1]
        and fields[2] == s1["gpu_uuid"]
        and fields[1] in s1["hardware"],
        "raw hardware differs from canonical S1 GPU identity",
    )
    for field in ("nvidia_smi", "nvcc", "cargo", "rustc"):
        require(isinstance(hardware[field], str) and hardware[field], f"hardware.{field} is empty")
    tools = require_dict(hardware["tools"], "hardware.tools")
    require(
        set(tools) == {"nvidia_smi", "nvcc", "cargo", "rustc"},
        "hardware tool set mismatch",
    )
    for name, identity in tools.items():
        row = require_dict(identity, f"hardware.tools.{name}")
        require_exact_fields(row, FILE_IDENTITY_FIELDS, f"hardware.tools.{name}")
        require(isinstance(row["path"], str) and Path(row["path"]).is_absolute(), f"hardware tool {name} path invalid")
        require_sha(row["sha256"], f"hardware tool {name} SHA256")
        require_int(row["size_bytes"], f"hardware tool {name} size")
    return hardware


def validate_raw_artifact_index(root: Path, manifest: dict[str, Any]) -> None:
    rows = require_list(manifest["artifacts"], "raw artifact index")
    require(manifest["artifact_count"] == len(rows), "raw artifact count mismatch")
    indexed_paths: list[str] = []
    for index, raw_row in enumerate(rows):
        label = f"raw artifacts[{index}]"
        row = require_dict(raw_row, label)
        require_exact_fields(row, FILE_IDENTITY_FIELDS, label)
        path = resolve_artifact(root, row["path"], label)
        validate_file_identity(path, row, label)
        indexed_paths.append(row["path"])
    require(len(indexed_paths) == len(set(indexed_paths)), "raw artifact index contains duplicates")
    actual: list[str] = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"raw artifact contains symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in {"raw.manifest.json", "failure.json"}:
                actual.append(relative)
    require(not (root / "failure.json").exists(), "ready raw artifact contains failure.json")
    require(actual == indexed_paths, "raw artifact index has missing or extra files")


def validate_plan_and_receipt(
    *,
    raw_root: Path,
    receipt_path: Path,
    source_root: Path,
    expected_command: list[str],
    step_id: str,
    max_processes: int,
    max_group_threads: int,
    max_per_process_threads: int,
) -> dict[str, Any]:
    step_root = receipt_path.parent
    require(
        step_root == raw_root / step_id
        and receipt_path.name == "bounded.receipt.json",
        f"{step_id} receipt is not at {step_id}/bounded.receipt.json",
    )
    plan_path = step_root / "plan.json"
    plan, _ = read_json(plan_path, "catalog export plan")
    plan_value = require_dict(plan, "catalog export plan")
    require_exact_fields(plan_value, PLAN_FIELDS, "catalog export plan")
    require(
        plan_value["schema_version"] == 1
        and plan_value["step_id"] == step_id
        and plan_value["command"] == expected_command
        and Path(str(plan_value["cwd"])).resolve() == source_root
        and isinstance(plan_value["progress_signal"], str)
        and plan_value["progress_signal"].strip(),
        "catalog export plan identity/command/cwd mismatch",
    )
    expected_duration = require_int(
        plan_value["expected_duration_seconds"], "catalog export expected duration", minimum=1
    )
    deadline = require_int(
        plan_value["hard_deadline_seconds"], "catalog export hard deadline", minimum=expected_duration
    )
    workers = require_dict(plan_value["worker_limits"], "catalog export worker limits")
    require(
        set(workers) == {"max_processes", "max_group_threads", "max_per_process_threads"},
        "catalog export worker-limit shape mismatch",
    )
    require(
        1 <= require_int(workers["max_processes"], "max_processes", minimum=1) <= max_processes
        and 1 <= require_int(workers["max_group_threads"], "max_group_threads", minimum=1)
        <= max_group_threads
        and 1 <= require_int(
            workers["max_per_process_threads"], "max_per_process_threads", minimum=1
        )
        <= max_per_process_threads,
        "catalog export worker bounds are not independently small",
    )

    receipt, _ = read_json(receipt_path, "catalog export bounded receipt")
    value = require_dict(receipt, "catalog export bounded receipt")
    require_exact_fields(value, RECEIPT_FIELDS, "catalog export bounded receipt")
    require(
        value["schema"] == BOUNDED_RECEIPT_SCHEMA
        and value["command"] == expected_command
        and Path(str(value["cwd"])).resolve() == source_root
        and value["status"] == "pass"
        and value["reason"] == "command_completed"
        and value["rc"] == 0
        and value["violation"] is None
        and value["sampling_error_count"] == 0
        and value["sampling_errors"] == []
        and value["termination"] == {"signals": [], "errors": []}
        and value["cleanup"] == {"process_group_gone": True},
        "catalog export bounded receipt did not pass cleanly",
    )
    require_int(value["pid"], "catalog export pid", minimum=1)
    require_int(value["pgid"], "catalog export pgid", minimum=1)
    require_int(value["successful_samples"], "catalog export successful samples")
    require(
        isinstance(value["duration_seconds"], (int, float))
        and not isinstance(value["duration_seconds"], bool)
        and 0 <= value["duration_seconds"] <= deadline + 5,
        "catalog export duration is invalid",
    )
    limits = require_dict(value["limits"], "catalog export receipt limits")
    require_exact_fields(limits, LIMIT_FIELDS, "catalog export receipt limits")
    require(
        limits["wall_timeout_seconds"] == float(deadline)
        and limits["max_processes"] == workers["max_processes"]
        and limits["max_group_threads"] == workers["max_group_threads"]
        and limits["max_per_process_threads"] == workers["max_per_process_threads"],
        "catalog export receipt limits differ from plan",
    )
    peaks = require_dict(value["peaks"], "catalog export peaks")
    require_exact_fields(peaks, PEAK_FIELDS, "catalog export peaks")
    for field, limit_field in (
        ("processes", "max_processes"),
        ("group_threads", "max_group_threads"),
        ("per_process_threads", "max_per_process_threads"),
    ):
        peak = require_int(peaks[field], f"catalog export peaks.{field}")
        require(peak <= limits[limit_field], f"catalog export peak {field} exceeds limit")
    stdout_path = step_root / "stdout.log"
    stderr_path = step_root / "stderr.log"
    for stream, local_path in (("stdout", stdout_path), ("stderr", stderr_path)):
        identity = require_dict(value[stream], f"receipt.{stream}")
        validate_file_identity(local_path, identity, f"receipt.{stream}")
        recorded = Path(str(identity["path"]))
        require(
            recorded.name == local_path.name and recorded.parent.name == step_id,
            f"receipt.{stream}.path does not identify the {step_id} log",
        )
    return {
        "plan": external_ref(plan_path),
        "receipt": external_ref(receipt_path),
        "stdout": external_ref(stdout_path),
        "stderr": external_ref(stderr_path),
        "deadline_seconds": deadline,
        "worker_limits": workers,
    }


def portable_command(
    command: Sequence[str],
    *,
    source_root: Path,
    raw_root: Path,
    target_dir: Path,
    native_build_cache: Path,
    native_import_dirs: Sequence[Path],
) -> list[str]:
    replacements = [
        (str(source_root), "<source-root>"),
        (str(raw_root), "<artifact-root>"),
        (str(target_dir), "<target-dir>"),
        (str(native_build_cache), "<native-build-cache>"),
    ]
    replacements.extend(
        (str(path), f"<native-import-dir-{index}>")
        for index, path in enumerate(native_import_dirs)
    )
    replacements.sort(key=lambda row: len(row[0]), reverse=True)
    result: list[str] = []
    for argument in command:
        rendered = argument
        for raw, replacement in replacements:
            rendered = rendered.replace(raw, replacement)
        result.append(rendered)
    return result


def expected_core_ptx_artifacts(source_root: Path) -> set[str]:
    build_rs = source_root / "crates/ferrum-kernels/build.rs"
    source = read_regular_bytes(build_rs, MAX_LOG_BYTES, "ferrum-kernels build.rs").decode(
        "utf-8"
    )
    match = CORE_PTX_BLOCK_RE.search(source)
    require(match is not None, "cannot parse CORE_PTX_KERNELS from ferrum-kernels/build.rs")
    assert match is not None
    paths = QUOTED_RUST_PATH_RE.findall(match.group("body"))
    require(paths and len(paths) == len(set(paths)), "CORE_PTX_KERNELS is empty or duplicated")
    return {f"core-ptx:{path}" for path in paths}


def cuda_build_inputs_hash(value: str) -> str:
    return f"sha256:{sha256_bytes(value.encode('utf-8'))}"


def validate_bootstrap_native_operator_set(
    raw_root: Path, value: Any
) -> tuple[dict[str, Any], Path]:
    bootstrap = require_dict(value, "raw bootstrap native operator set")
    require_exact_fields(
        bootstrap, RAW_BOOTSTRAP_FIELDS, "raw bootstrap native operator set"
    )
    require(
        bootstrap["role"]
        == "build bootstrap only; G07B must rebuild artifacts against the exported live catalog",
        "raw bootstrap role changed",
    )
    lock_path = validate_relative_ref(
        raw_root, bootstrap["lock"], "raw bootstrap native operator lock"
    )
    try:
        validated = validate_native_operator_set(
            lock_path, REQUIRED_CUDA_NATIVE_OPERATORS
        )
    except NativeOperatorSetEvidenceError as error:
        raise CheckpointError(str(error)) from error
    expected_identity = native_operator_set_public_identity(validated)
    actual_identity = {
        key: bootstrap[key] for key in NATIVE_OPERATOR_SET_PUBLIC_IDENTITY_FIELDS
    }
    require(
        actual_identity == expected_identity,
        "raw bootstrap native operator public identity differs from its lock closure",
    )
    require(
        bootstrap["lock"]["sha256"] == expected_identity["lock_sha256"]
        and bootstrap["lock"]["size_bytes"] == expected_identity["lock_size_bytes"],
        "raw bootstrap lock evidence differs from its public identity",
    )
    return bootstrap, lock_path


def parse_build_command(command: Any) -> tuple[list[str], dict[str, str], str, list[str]]:
    parts = require_list(command, "raw build command")
    require(
        len(parts) >= 4 and all(isinstance(part, str) and part for part in parts),
        "raw build command is invalid",
    )
    require(parts[0] == "/usr/bin/env", "raw build command must use /usr/bin/env")
    environment: dict[str, str] = {}
    index = 1
    while index < len(parts) and "=" in parts[index]:
        key, raw_value = parts[index].split("=", 1)
        require(
            re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is not None
            and key not in environment,
            f"raw build command has invalid environment assignment: {parts[index]!r}",
        )
        environment[key] = raw_value
        index += 1
    require(index < len(parts), "raw build command has no executable")
    return parts, environment, parts[index], parts[index + 1 :]


def validate_build_summary(
    path: Path,
    *,
    source_root: Path,
    build_lock_path: Path,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    raw, _ = read_json(path, "raw CUDA build summary")
    value = require_dict(raw, "raw CUDA build summary")
    require_exact_fields(
        value,
        {"schema_version", "artifact_type", "rows"},
        "raw CUDA build summary",
    )
    require(
        value["schema_version"] == 1
        and value["artifact_type"] == "ferrum_cuda_build_summary_receipt",
        "raw CUDA build summary schema mismatch",
    )
    rows = require_list(value["rows"], "raw CUDA build summary rows")
    by_artifact: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = require_dict(raw_row, f"raw CUDA build summary row[{index}]")
        require_exact_fields(
            row,
            {"artifact", "status", "reason", "elapsed_ms", "inputs_hash"},
            f"raw CUDA build summary row[{index}]",
        )
        artifact = row["artifact"]
        require(
            isinstance(artifact, str)
            and artifact
            and artifact not in by_artifact
            and isinstance(row["status"], str)
            and row["status"] not in {"failed", "rejected"}
            and isinstance(row["reason"], str)
            and row["reason"]
            and isinstance(row["elapsed_ms"], int)
            and not isinstance(row["elapsed_ms"], bool)
            and row["elapsed_ms"] >= 0
            and re.fullmatch(r"sha256:[0-9a-f]{64}", str(row["inputs_hash"]))
            is not None,
            f"raw CUDA build summary row[{index}] is malformed",
        )
        by_artifact[artifact] = row
    expected_core = expected_core_ptx_artifacts(source_root)
    expected_non_core = {
        "native_operator_artifact_set",
        *(unit[0] for unit in REQUIRED_CUDA_NATIVE_BUILD_UNITS),
    }
    require(
        set(by_artifact) == expected_core | expected_non_core,
        "raw CUDA build summary build-unit denominator mismatch",
    )
    require(
        all(
            by_artifact[name]["status"] == "cache_hit"
            and by_artifact[name]["reason"] == "signature-match"
            for name in expected_core
        ),
        "raw live-catalog build compiled or reused stale core PTX",
    )
    binary_rows = bootstrap["binary_sha256_by_operator"]
    binaries = ",".join(
        f"{row['operator']}={row['sha256']}" for row in binary_rows
    )
    units = ",".join(unit[1] for unit in REQUIRED_CUDA_NATIVE_BUILD_UNITS)
    signature = (
        f"lock={build_lock_path}:"
        f"lock_sha256={bootstrap['lock']['sha256']}:"
        f"catalog={bootstrap['g03_catalog_sha256']}:"
        f"operators={len(REQUIRED_CUDA_NATIVE_BUILD_UNITS)}:"
        f"operator_binaries={binaries}:build_units={units}"
    )
    set_row = by_artifact["native_operator_artifact_set"]
    require(
        set_row["status"] == "linked"
        and set_row["reason"] == "manifest-v3-artifact-set-v5-validated"
        and set_row["inputs_hash"] == cuda_build_inputs_hash(signature),
        "raw CUDA build did not bind the staged native operator set",
    )
    for summary_artifact, _unit_name, operator in REQUIRED_CUDA_NATIVE_BUILD_UNITS:
        row = by_artifact[summary_artifact]
        require(
            row["status"] == "artifact"
            and row["reason"] == "native-operator-artifact-set"
            and row["inputs_hash"] == cuda_build_inputs_hash(operator),
            f"raw CUDA build did not bind {summary_artifact}",
        )
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": len(rows),
        "native_operator_artifact_set_status": "linked",
        "native_operator_artifact_set_inputs_hash": set_row["inputs_hash"],
        "core_ptx_count": len(expected_core),
        "core_ptx_status": "cache_hit",
        "core_ptx_artifacts_sha256": canonical_json_sha256(sorted(expected_core)),
    }


def validate_build(
    *,
    raw_root: Path,
    source_root: Path,
    scope: dict[str, Any],
    bootstrap: dict[str, Any],
    lock_path: Path,
    value: Any,
) -> dict[str, Any]:
    build = require_dict(value, "raw build")
    require_exact_fields(build, RAW_BUILD_FIELDS, "raw build")
    command, environment, cargo, arguments = parse_build_command(build["command"])
    expected_environment = {
        "PATH",
        "RUSTC",
        "CARGO_TARGET_DIR",
        "CARGO_BUILD_JOBS",
        "CUDA_COMPUTE_CAP",
        "FERRUM_NVCC_THREADS",
        "FERRUM_CUDA_NATIVE_BUILD_CACHE",
        "FERRUM_CUDA_NATIVE_IMPORT_DIRS",
        "FERRUM_NATIVE_OPERATOR_SET_LOCK",
        "FERRUM_CUDA_NATIVE_SOURCE_POLICY",
        "FERRUM_CUDA_BUILD_SUMMARY_RECEIPT",
    }
    require(set(environment) == expected_environment, "raw build environment field set mismatch")
    require(
        Path(cargo).name == "cargo"
        and Path(environment["RUSTC"]).name == "rustc"
        and environment["CARGO_BUILD_JOBS"] == str(scope["cargo_jobs"])
        and environment["CUDA_COMPUTE_CAP"] == "89"
        and environment["FERRUM_NVCC_THREADS"] == "4"
        and environment["FERRUM_CUDA_NATIVE_SOURCE_POLICY"] == "cache-only"
        and arguments
        == [
            "build",
            "--profile",
            "cuda-correctness",
            "--locked",
            "--jobs",
            str(scope["cargo_jobs"]),
            "-p",
            "ferrum-kernels",
            "--example",
            "runtime_vnext_cuda_catalog_input",
            "--features",
            "cuda,vllm-moe-marlin,vllm-paged-attn-v2",
        ],
        "raw build command does not match the canonical catalog exporter build",
    )
    target_dir = Path(environment["CARGO_TARGET_DIR"]).expanduser().resolve()
    native_cache = Path(str(build["native_build_cache"])).expanduser().resolve()
    import_values = require_list(build["native_import_dirs"], "raw native import dirs")
    require(
        all(isinstance(path, str) and Path(path).is_absolute() for path in import_values),
        "raw native import dirs are invalid",
    )
    native_imports = [Path(path).expanduser().resolve() for path in import_values]
    require(
        environment["FERRUM_CUDA_NATIVE_BUILD_CACHE"] == str(native_cache)
        and environment["FERRUM_CUDA_NATIVE_IMPORT_DIRS"]
        == os.pathsep.join(str(path) for path in native_imports)
        and Path(environment["FERRUM_NATIVE_OPERATOR_SET_LOCK"]).resolve()
        == lock_path.resolve(),
        "raw build cache/import/lock contract mismatch",
    )
    summary_path = validate_relative_ref(
        raw_root, build["summary_receipt"], "raw build summary receipt"
    )
    require(
        Path(environment["FERRUM_CUDA_BUILD_SUMMARY_RECEIPT"]).resolve()
        == summary_path.resolve(),
        "raw build summary command path mismatch",
    )
    summary = validate_build_summary(
        summary_path,
        source_root=source_root,
        build_lock_path=lock_path,
        bootstrap=bootstrap,
    )
    declared_summary = require_dict(build["summary"], "raw build summary identity")
    require_exact_fields(
        declared_summary, RAW_BUILD_SUMMARY_FIELDS, "raw build summary identity"
    )
    require(declared_summary == summary, "raw build summary identity is stale")
    receipt_path = validate_relative_ref(raw_root, build["receipt"], "raw build receipt")
    bounded = validate_plan_and_receipt(
        raw_root=raw_root,
        receipt_path=receipt_path,
        source_root=source_root,
        expected_command=command,
        step_id="build",
        max_processes=16,
        max_group_threads=192,
        max_per_process_threads=64,
    )
    expected_portable = portable_command(
        command,
        source_root=source_root,
        raw_root=raw_root,
        target_dir=target_dir,
        native_build_cache=native_cache,
        native_import_dirs=native_imports,
    )
    require(build["portable_command"] == expected_portable, "raw build portable command drifted")
    return {"bounded": bounded, "summary": summary}


def raw_summary_from_catalogs(
    provider: dict[str, Any], capability: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "sha256": provider["file_sha256"],
            "size_bytes": provider["size_bytes"],
            "canonical_fingerprint": provider["collector_fingerprint"],
            "provider_count": provider["provider_count"],
            "operation_count": provider["operation_count"],
            "provider_ids": provider["provider_ids"],
            "operation_ids": provider["operation_ids"],
        },
        {
            "sha256": capability["file_sha256"],
            "size_bytes": capability["size_bytes"],
            "canonical_fingerprint": capability["collector_fingerprint"],
            "device_id": capability["device_id"],
            "runtime_implementation_fingerprint": capability[
                "runtime_implementation_fingerprint"
            ],
            "operations_count": capability["operations_count"],
            "providers_count": capability["providers_count"],
            "engine_providers_count": capability["engine_providers_count"],
            "weight_materializers_count": capability["weight_materializers_count"],
        },
    )


def validate_raw_collection(
    *,
    source_root: Path,
    source: dict[str, Any],
    s1: dict[str, Any],
    provider_path: Path,
    capability_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    raw_root = receipt_path.parent.parent.resolve()
    require(raw_root.is_dir() and not raw_root.is_symlink(), "raw artifact root is invalid")
    require(not raw_root.is_relative_to(source_root), "raw artifact root must be outside source")
    require(
        provider_path.resolve() == raw_root / "catalog/provider-catalog.json",
        "provider catalog path is not canonical",
    )
    require(
        capability_path.resolve() == raw_root / "catalog/capability-catalog.json",
        "capability catalog path is not canonical",
    )

    provider_raw = read_regular_bytes(provider_path, MAX_JSON_BYTES, "provider catalog")
    capability_raw = read_regular_bytes(capability_path, MAX_JSON_BYTES, "capability catalog")
    provider = validate_provider_catalog_bytes(provider_raw)
    capability = validate_capability_catalog_bytes(capability_raw, cuda_ordinal=0)
    require(
        provider["value"]["providers"] == capability["projection"],
        "provider catalog is not the exact projection of the capability catalog",
    )
    require(
        provider["operation_ids"] == capability["operation_ids"],
        "provider/capability operation denominators differ",
    )

    raw_manifest_path = raw_root / "raw.manifest.json"
    raw_manifest, raw_manifest_bytes = read_json(raw_manifest_path, "raw collection manifest")
    manifest = require_dict(raw_manifest, "raw collection manifest")
    require_exact_fields(manifest, RAW_MANIFEST_FIELDS, "raw collection manifest")
    require(
        manifest["schema_version"] == 1
        and manifest["artifact_type"] == RAW_ARTIFACT_TYPE
        and manifest["status"] == "ready"
        and manifest["source"] == source
        and manifest["does_not_prove"] == RAW_DOES_NOT_PROVE,
        "raw collection manifest identity/source/proof boundary mismatch",
    )
    source_path = raw_root / "source.json"
    hardware_path = raw_root / "hardware.json"
    raw_source, _ = read_json(source_path, "raw source identity")
    raw_hardware, _ = read_json(hardware_path, "raw hardware identity")
    require(raw_source == source and manifest["source"] == raw_source, "raw source identity drifted")
    hardware = validate_hardware(raw_hardware, s1)
    require(manifest["hardware"] == hardware, "raw hardware differs from manifest")

    collector_ref = require_dict(manifest["collector"], "raw collector ref")
    require_exact_fields(collector_ref, RAW_COLLECTOR_FIELDS, "raw collector ref")
    collector_path = source_root / "scripts/release/runtime_vnext_g03_live_catalog_collect.py"
    require(
        collector_ref["source_path"]
        == "scripts/release/runtime_vnext_g03_live_catalog_collect.py",
        "raw collector source path mismatch",
    )
    collector_identity = {
        key: collector_ref[key] for key in ("path", "sha256", "size_bytes")
    }
    collector_snapshot = validate_relative_ref(
        raw_root, collector_identity, "raw collector snapshot"
    )
    require(
        collector_snapshot == raw_root / "source-snapshot" / collector_path.name
        and sha256_file(collector_path) == collector_ref["sha256"]
        and collector_path.stat().st_size == collector_ref["size_bytes"],
        "raw collector snapshot differs from the clean source collector",
    )
    scope = require_dict(manifest["scope"], "raw scope")
    require_exact_fields(scope, RAW_SCOPE_FIELDS, "raw scope")
    require(
        scope["backend"] == "cuda"
        and scope["gpu_count"] == 1
        and scope["gpu_model"] == "RTX 4090"
        and scope["cuda_ordinal"] == 0
        and scope["attention_policy"] == "native-adaptive"
        and scope["cargo_profile"] == "cuda-correctness"
        and isinstance(scope["cargo_jobs"], int)
        and 1 <= scope["cargo_jobs"] <= 8
        and scope["features"]
        == ["cuda", "vllm-moe-marlin", "vllm-paged-attn-v2"],
        "raw scope is not the canonical S1 live-catalog lane",
    )
    bootstrap, lock_path = validate_bootstrap_native_operator_set(
        raw_root, manifest["bootstrap_native_operator_set"]
    )
    build = validate_build(
        raw_root=raw_root,
        source_root=source_root,
        scope=scope,
        bootstrap=bootstrap,
        lock_path=lock_path,
        value=manifest["build"],
    )

    export = require_dict(manifest["export"], "raw export")
    require_exact_fields(export, RAW_EXPORT_FIELDS, "raw export")
    command = require_list(export["command"], "raw export command")
    require(
        len(command) == 5
        and all(isinstance(part, str) and part for part in command)
        and Path(command[0]).name == "runtime_vnext_cuda_catalog_input"
        and command[1:] == [
            "0",
            "native-adaptive",
            str(provider_path.resolve()),
            str(capability_path.resolve()),
        ],
        "raw export command is not the exact live-catalog exporter command",
    )
    binary_path = validate_relative_ref(raw_root, export["binary"], "raw export binary")
    require(binary_path.resolve() == Path(command[0]).resolve(), "raw export command binary mismatch")
    require(os.access(binary_path, os.X_OK), "raw catalog exporter binary is not executable")
    receipt_ref_path = validate_relative_ref(raw_root, export["receipt"], "raw export receipt")
    require(receipt_ref_path == receipt_path.resolve(), "raw export receipt path mismatch")
    require(export["receipt_status"] == "pass", "raw export receipt status is not pass")
    bounded = validate_plan_and_receipt(
        raw_root=raw_root,
        receipt_path=receipt_path,
        source_root=source_root,
        expected_command=command,
        step_id="catalog-export",
        max_processes=2,
        max_group_threads=32,
        max_per_process_threads=32,
    )
    target_dir = Path(
        parse_build_command(manifest["build"]["command"])[1]["CARGO_TARGET_DIR"]
    ).resolve()
    native_cache = Path(str(manifest["build"]["native_build_cache"])).resolve()
    native_imports = [
        Path(str(path)).resolve() for path in manifest["build"]["native_import_dirs"]
    ]
    require(
        export["portable_command"]
        == portable_command(
            command,
            source_root=source_root,
            raw_root=raw_root,
            target_dir=target_dir,
            native_build_cache=native_cache,
            native_import_dirs=native_imports,
        ),
        "raw export portable command drifted",
    )

    provider_summary, capability_summary = raw_summary_from_catalogs(provider, capability)
    provider_export = require_dict(export["provider_catalog"], "raw provider export")
    capability_export = require_dict(export["capability_catalog"], "raw capability export")
    require(
        provider_export == {"path": "catalog/provider-catalog.json", **provider_summary},
        "raw provider catalog summary is stale",
    )
    require(
        capability_export
        == {"path": "catalog/capability-catalog.json", **capability_summary},
        "raw capability catalog summary is stale",
    )
    stdout = read_regular_bytes(raw_root / "catalog-export/stdout.log", MAX_LOG_BYTES, "catalog export stdout")
    try:
        stdout_text = stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CheckpointError(f"catalog export stdout is not UTF-8: {error}") from error
    ready_lines = [line for line in stdout_text.splitlines() if line.startswith(READY_PREFIX)]
    expected_ready = (
        f"{READY_PREFIX} provider={provider_path.resolve()} capability={capability_path.resolve()} "
        f"provider_count={provider['provider_count']} "
        f"capability_fingerprint={capability['runtime_fingerprint']}"
    )
    require(ready_lines == [expected_ready], "catalog export READY line is missing, duplicated, or stale")
    readiness = require_dict(export["readiness"], "raw export readiness")
    require_exact_fields(readiness, RAW_READINESS_FIELDS, "raw export readiness")
    require(
        readiness
        == {
            "line": expected_ready,
            "provider_count": provider["provider_count"],
            "capability_fingerprint": capability["runtime_fingerprint"],
        },
        "raw export readiness summary is stale",
    )
    validate_raw_artifact_index(raw_root, manifest)
    return {
        "root": str(raw_root),
        "manifest": {
            "path": str(raw_manifest_path),
            "sha256": sha256_bytes(raw_manifest_bytes),
            "size_bytes": len(raw_manifest_bytes),
        },
        "source": external_ref(source_path),
        "hardware": external_ref(hardware_path),
        "collector": external_ref(collector_snapshot),
        "binary": external_ref(binary_path),
        "bounded_build": build,
        "bounded_export": bounded,
        "scope": scope,
        "provider": provider,
        "capability": capability,
        "provider_bytes": provider_raw,
        "capability_bytes": capability_raw,
    }


def checkpoint_artifact_index(root: Path) -> list[dict[str, Any]]:
    excluded = {
        "manifest.json",
        "gate.manifest.json",
        "run_gate.child.command.json",
        "run_gate.child.stdout",
        "run_gate.child.stderr",
    }
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"checkpoint artifact contains symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(file_identity(path, relative_to=root))
    return rows


def canonical_index_sha256(rows: list[dict[str, Any]]) -> str:
    return canonical_json_sha256(rows)


def copied_ref(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    return file_identity(path, relative_to=root)


def acceptance_summary(raw: dict[str, Any]) -> dict[str, Any]:
    provider = raw["provider"]
    capability = raw["capability"]
    return {
        "clean_source": True,
        "s1_source_and_binary_bound": True,
        "bounded_export_pass": True,
        "hardware_and_attention_policy_bound": True,
        "provider_catalog_canonical": True,
        "capability_catalog_canonical": True,
        "provider_projection_exact": True,
        "provider_count": provider["provider_count"],
        "operation_count": provider["operation_count"],
        "capability_provider_row_count": capability["provider_row_count"],
        "catalog_mismatch_count": 0,
    }


def build_manifest(
    *,
    output: Path,
    artifact_root: Path,
    source: dict[str, Any],
    s1: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    artifacts = {
        "provider_catalog": copied_ref(artifact_root, "provider-catalog.json"),
        "capability_catalog": copied_ref(artifact_root, "capability-catalog.json"),
    }
    index = checkpoint_artifact_index(artifact_root)
    pass_line = f"{PASS_PREFIX}: {output}"
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "checkpoint_id": "G03-live-catalog",
        "lane": LANE,
        "status": "pass",
        "canonical": True,
        "artifact_dir": str(output),
        "source": source,
        "dependencies": {
            "s1": s1,
            "raw_collection": {
                key: raw[key]
                for key in (
                    "root",
                    "manifest",
                    "source",
                    "hardware",
                    "collector",
                    "binary",
                    "bounded_export",
                    "scope",
                )
            },
        },
        "catalogs": {
            "provider": {
                **artifacts["provider_catalog"],
                "schema_version": 1,
                "backend": "cuda",
                "canonical_sha256": raw["provider"]["file_sha256"],
                "provider_count": raw["provider"]["provider_count"],
                "operation_count": raw["provider"]["operation_count"],
            },
            "capability": {
                **artifacts["capability_catalog"],
                "device_id": raw["capability"]["device_id"],
                "runtime_implementation_fingerprint": raw["capability"][
                    "runtime_implementation_fingerprint"
                ],
                "capability_catalog_fingerprint": raw["capability"][
                    "runtime_fingerprint"
                ],
                "provider_row_count": raw["capability"]["provider_row_count"],
            },
        },
        "acceptance": acceptance_summary(raw),
        "artifacts": artifacts,
        "artifact_index": index,
        "artifact_index_sha256": canonical_index_sha256(index),
        "unlocks": ["G07B"],
        "does_not_prove": DOES_NOT_PROVE,
        "pass_line": pass_line,
    }


MANIFEST_FIELDS = {
    "schema_version",
    "artifact_type",
    "checkpoint_id",
    "lane",
    "status",
    "canonical",
    "artifact_dir",
    "source",
    "dependencies",
    "catalogs",
    "acceptance",
    "artifacts",
    "artifact_index",
    "artifact_index_sha256",
    "unlocks",
    "does_not_prove",
    "pass_line",
}


def verify_checkpoint_manifest(
    manifest_path: Path,
    *,
    source_root: Path,
    verify_checkout: bool,
    artifact_root_override: Path | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    root = (artifact_root_override or manifest_path.parent).resolve()
    manifest, _ = read_json(manifest_path, "G03 live catalog checkpoint manifest")
    value = require_dict(manifest, "G03 live catalog checkpoint manifest")
    require_exact_fields(value, MANIFEST_FIELDS, "G03 live catalog checkpoint manifest")
    declared_root = Path(str(value.get("artifact_dir", ""))).expanduser().resolve()
    if artifact_root_override is None:
        require(declared_root == root, "checkpoint artifact_dir mismatch")
    require(
        value["schema_version"] == SCHEMA_VERSION
        and value["artifact_type"] == ARTIFACT_TYPE
        and value["checkpoint_id"] == "G03-live-catalog"
        and value["lane"] == LANE
        and value["status"] == "pass"
        and value["canonical"] is True
        and value["unlocks"] == ["G07B"]
        and value["does_not_prove"] == DOES_NOT_PROVE
        and value["pass_line"] == f"{PASS_PREFIX}: {declared_root}",
        "checkpoint identity/status/PASS/proof boundary mismatch",
    )
    source = source_identity(source_root) if verify_checkout else value["source"]
    require(value["source"] == source, "checkpoint source differs from clean checkout")
    dependencies = require_dict(value["dependencies"], "checkpoint dependencies")
    require(set(dependencies) == {"s1", "raw_collection"}, "checkpoint dependency set mismatch")
    s1_manifest_path = validate_external_ref(
        require_dict(dependencies["s1"], "S1 dependency")["outer_manifest"],
        "S1 dependency outer manifest",
    )
    s1 = validate_s1_outer_manifest(s1_manifest_path, source)
    require(s1 == dependencies["s1"], "checkpoint S1 dependency summary drifted")
    raw_dep = require_dict(dependencies["raw_collection"], "raw collection dependency")
    require(
        set(raw_dep)
        == {
            "root",
            "manifest",
            "source",
            "hardware",
            "collector",
            "binary",
            "bounded_export",
            "scope",
        },
        "raw collection dependency shape mismatch",
    )
    raw_root = Path(str(raw_dep["root"])).expanduser().resolve()
    raw_manifest_path = validate_external_ref(raw_dep["manifest"], "raw collection manifest ref")
    require(raw_manifest_path == raw_root / "raw.manifest.json", "raw manifest root mismatch")
    receipt_path = validate_external_ref(
        require_dict(raw_dep["bounded_export"], "bounded export dependency")["receipt"],
        "bounded export receipt ref",
    )
    provider_path = root / "provider-catalog.json"
    capability_path = root / "capability-catalog.json"
    provider_copy = read_regular_bytes(provider_path, MAX_JSON_BYTES, "copied provider catalog")
    capability_copy = read_regular_bytes(capability_path, MAX_JSON_BYTES, "copied capability catalog")
    raw = validate_raw_collection(
        source_root=source_root,
        source=source,
        s1=s1,
        provider_path=raw_root / "catalog/provider-catalog.json",
        capability_path=raw_root / "catalog/capability-catalog.json",
        receipt_path=receipt_path,
    )
    require(
        provider_copy == raw["provider_bytes"]
        and capability_copy == raw["capability_bytes"],
        "checkpoint catalog copies differ from revalidated raw catalogs",
    )
    expected_raw_dep = {
        key: raw[key]
        for key in (
            "root",
            "manifest",
            "source",
            "hardware",
            "collector",
            "binary",
            "bounded_export",
            "scope",
        )
    }
    require(raw_dep == expected_raw_dep, "raw collection dependency summary drifted")
    artifacts = require_dict(value["artifacts"], "checkpoint artifacts")
    require(set(artifacts) == {"provider_catalog", "capability_catalog"}, "checkpoint artifact set mismatch")
    require(
        validate_relative_ref(root, artifacts["provider_catalog"], "provider artifact")
        == provider_path
        and validate_relative_ref(root, artifacts["capability_catalog"], "capability artifact")
        == capability_path,
        "checkpoint artifact paths are not canonical",
    )
    index = checkpoint_artifact_index(root)
    require(value["artifact_index"] == index, "checkpoint artifact index drifted")
    require(
        value["artifact_index_sha256"] == canonical_index_sha256(index),
        "checkpoint artifact index SHA256 mismatch",
    )
    catalogs = require_dict(value["catalogs"], "checkpoint catalogs")
    require(set(catalogs) == {"provider", "capability"}, "checkpoint catalog summary shape mismatch")
    expected = build_manifest(
        output=declared_root,
        artifact_root=root,
        source=source,
        s1=s1,
        raw=raw,
    )
    for field in ("catalogs", "acceptance", "artifacts", "artifact_index", "artifact_index_sha256"):
        require(value[field] == expected[field], f"checkpoint {field} summary drifted")
    return value


def write_bytes_create_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_create_new(path: Path, value: Any) -> None:
    write_bytes_create_new(
        path,
        (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii"),
    )


def execute(args: argparse.Namespace) -> str:
    source_root = args.source_root.expanduser().resolve()
    s1_path = args.s1_manifest.expanduser().resolve()
    provider_path = args.provider_catalog.expanduser().resolve()
    capability_path = args.capability_catalog.expanduser().resolve()
    receipt_path = args.catalog_export_receipt.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if output.exists():
        require(
            output.is_dir()
            and not output.is_symlink()
            and not any(output.iterdir()),
            f"checkpoint output must be absent or an empty real directory: {output}",
        )
        output.rmdir()
    require(not output.is_relative_to(source_root), "checkpoint output must be outside source root")
    source = source_identity(source_root)
    s1 = validate_s1_outer_manifest(s1_path, source)
    raw = validate_raw_collection(
        source_root=source_root,
        source=source,
        s1=s1,
        provider_path=provider_path,
        capability_path=capability_path,
        receipt_path=receipt_path,
    )
    require(source_identity(source_root) == source, "source changed while validating live catalog")
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.parent.is_symlink(), "checkpoint output parent must not be a symlink")
    staging = output.parent / f".{output.name}.{os.getpid()}.tmp"
    require(not staging.exists(), f"staging path already exists: {staging}")
    staging.mkdir()
    try:
        write_bytes_create_new(staging / "provider-catalog.json", raw["provider_bytes"])
        write_bytes_create_new(staging / "capability-catalog.json", raw["capability_bytes"])
        manifest = build_manifest(
            output=output,
            artifact_root=staging,
            source=source,
            s1=s1,
            raw=raw,
        )
        write_json_create_new(staging / "manifest.json", manifest)
        verify_checkpoint_manifest(
            staging / "manifest.json",
            source_root=source_root,
            verify_checkout=True,
            artifact_root_override=staging,
        )
        require(not output.exists(), f"checkpoint output appeared during validation: {output}")
        os.replace(staging, output)
        verify_checkpoint_manifest(
            output / "manifest.json",
            source_root=source_root,
            verify_checkout=True,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)
        raise
    return f"{PASS_PREFIX}: {output}"


def expect_reject(action: Callable[[], Any], label: str) -> None:
    try:
        action()
    except CheckpointError:
        return
    raise CheckpointError(f"self-test mutation unexpectedly passed: {label}")


def fixture_source(root: Path) -> tuple[Path, dict[str, Any]]:
    source = root / "source"
    (source / "scripts/release").mkdir(parents=True)
    (source / "crates/ferrum-kernels").mkdir(parents=True)
    (source / "Cargo.toml").write_text("[workspace]\nmembers = []\n", encoding="utf-8")
    (source / "crates/ferrum-kernels/build.rs").write_text(
        'const CORE_PTX_KERNELS: &[&str] = &["kernels/fixture.cu"];\n',
        encoding="utf-8",
    )
    collector = source / "scripts/release/runtime_vnext_g03_live_catalog_collect.py"
    collector.write_text("#!/usr/bin/env python3\n# fixture collector\n", encoding="utf-8")
    run_text(source, ["git", "init", "-q"])
    run_text(source, ["git", "config", "user.email", "fixture@example.invalid"])
    run_text(source, ["git", "config", "user.name", "Fixture"])
    run_text(source, ["git", "add", "."])
    run_text(source, ["git", "commit", "-q", "-m", "fixture"])
    return source, source_identity(source)


def fixture_semantics() -> dict[str, Any]:
    version = {"major": 1, "minor": 0}
    repeatability = "bitwise_same_runtime"
    replay = "bitwise_eager_equivalent"
    fingerprint = execution_semantics_fingerprint((1, 0), repeatability, replay)
    return {
        "contract_version": dict(version),
        "contract_fingerprint": fingerprint,
        "repeatability": repeatability,
        "replay_equivalence": replay,
    }


def fixture_catalogs() -> tuple[dict[str, Any], dict[str, Any]]:
    version = {"major": 1, "minor": 0}
    profile = {"allocator": "linear_arena", "view": "contiguous"}
    operation = {
        "id": "operation.fixture",
        "version": dict(version),
        "inputs": [],
        "outputs": [
            {
                "dimensions": [{"exact": 1}],
                "element_types": ["f32"],
                "layouts": ["contiguous"],
                "access": "write",
                "alias": "no_alias",
            }
        ],
        "attributes": {"entries": {}},
        "resources": {
            "minimum_value_alignment_bytes": 16,
            "scratch": "optional",
            "binding": "optional",
            "persistent": "forbidden",
        },
        "oracle": "exact",
        "provider": {
            "minimum_version": dict(version),
            "required_capabilities": ["capability.fixture"],
        },
        "profile_phase": "forward",
    }
    operation_fingerprint = sha256_bytes(rust_compact_json_bytes(operation))
    provider_descriptor = {
        "provider_id": "provider.cuda.fixture",
        "operation_id": "operation.fixture",
        "operation_fingerprint": operation_fingerprint,
        "provider_implementation_fingerprint": "2" * 64,
        "execution_semantics": fixture_semantics(),
        "version": dict(version),
        "device_id": "cuda:0",
        "capabilities": ["capability.fixture"],
        "accepted_weight_formats": [],
        "accepted_quantization_formats": [],
        "dynamic_storage_bindings": [
            {
                "role": "output",
                "ordinal": 0,
                "storage": {"accepted_profiles": [dict(profile)]},
            }
        ],
        "resource_estimator_id": "resource-estimator.fixture",
        "resource_estimator_version": dict(version),
        "resource_estimator_implementation_fingerprint": "4" * 64,
    }
    capability = {
        "device": {
            "id": "cuda:0",
            "class": "accelerator",
            "ordinal": 0,
            "total_memory_bytes": 24 * 1024 * 1024 * 1024,
            "runtime_implementation_fingerprint": "3" * 64,
            "capabilities": ["capability.fixture"],
            "dynamic_storage_profiles": [dict(profile)],
        },
        "operations": {"operation.fixture": operation},
        "providers": {"operation.fixture": [provider_descriptor]},
        "engine_providers": {
            "provider.cuda.engine": {
                "provider_id": "provider.cuda.engine",
                "contract_version": dict(version),
                "implementation_fingerprint": "5" * 64,
                "device_id": "cuda:0",
                "capabilities": ["capability.fixture"],
            }
        },
        "weight_materializers": {
            "weight-materializer.identity": {
                "id": "weight-materializer.identity",
                "version": {"major": 2, "minor": 0},
                "implementation_fingerprint": identity_materializer_fingerprint(),
                "fidelity": "exact",
                "required_capabilities": [],
            }
        },
    }
    provider = {
        "schema_version": 1,
        "backend": "cuda",
        "providers": [
            {
                "operation_id": "operation.fixture",
                "operation_contract_version": dict(version),
                "operation_fingerprint": operation_fingerprint,
                "provider_id": "provider.cuda.fixture",
                "provider_version": dict(version),
                "provider_implementation_fingerprint": "2" * 64,
            }
        ],
    }
    return provider, capability


def fixture_s1(root: Path, source: dict[str, Any], gpu_uuid: str) -> Path:
    s1 = (root / "s1").resolve()
    s1.mkdir()
    binary_sha = "a" * 64
    hardware = "NVIDIA GeForce RTX 4090, 24564 MiB"
    validation = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_s1_cuda_basic_slice_validation",
        "status": "pass",
        "source_git_sha": source["git_sha"],
        "binary_sha256": binary_sha,
        "hardware": hardware,
    }
    validation_path = s1 / "validation.json"
    write_json_create_new(validation_path, validation)
    manifest = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_s1_cuda_basic_slice_manifest",
        "checkpoint_id": "S1-CUDA-basic",
        "lane": "runtime-vnext-s1-cuda",
        "status": "pass",
        "pass_line": f"FERRUM RUNTIME VNEXT S1 CUDA BASIC SLICE PASS: {s1}",
        "artifact_dir": str(s1),
        "source_git_sha": source["git_sha"],
        "binary_sha256": binary_sha,
        "hardware": hardware,
        "backend": "cuda",
        "entrypoints": ["ferrum run", "ferrum serve"],
        "validation": file_identity(validation_path, relative_to=s1),
        "metrics": {"gpu_uuid": gpu_uuid},
    }
    manifest_path = s1 / "manifest.json"
    write_json_create_new(manifest_path, manifest)
    outer = {
        "schema_version": 1,
        "lane": "vnext-s1-cuda",
        "status": "pass",
        "child_returncode": 0,
        "child_pass_line": manifest["pass_line"],
        "child_artifacts": {
            "kind": "delegated-manifest",
            "child_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
        },
        "git_sha": source["git_sha"],
        "dirty_status": {"is_dirty": False, "status_short": []},
        "artifact_dir": str(s1),
        "pass_line": f"FERRUM GATE vnext-s1-cuda PASS: {s1}",
        "error": None,
    }
    outer_path = s1 / "gate.manifest.json"
    write_json_create_new(outer_path, outer)
    return outer_path


def fixture_receipt(
    root: Path,
    source_root: Path,
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    *,
    wall_timeout_seconds: float = 30.0,
    max_processes: int = 2,
    max_group_threads: int = 32,
    max_per_process_threads: int = 32,
) -> dict[str, Any]:
    return {
        "schema": BOUNDED_RECEIPT_SCHEMA,
        "command": command,
        "cwd": str(source_root),
        "pid": 101,
        "pgid": 101,
        "limits": {
            "wall_timeout_seconds": wall_timeout_seconds,
            "max_processes": max_processes,
            "max_group_threads": max_group_threads,
            "max_per_process_threads": max_per_process_threads,
            "sample_interval_seconds": 0.2,
            "max_sampling_errors": 3,
            "term_grace_seconds": 3.0,
        },
        "peaks": {
            "processes": min(1, max_processes),
            "group_threads": min(2, max_group_threads),
            "per_process_threads": min(2, max_per_process_threads),
            "per_process_threads_pid": 101,
        },
        "started_at": "2026-08-05T00:00:00.000Z",
        "ended_at": "2026-08-05T00:00:01.000Z",
        "duration_seconds": 1.0,
        "reason": "command_completed",
        "rc": 0,
        "status": "pass",
        "successful_samples": 1,
        "sampling_error_count": 0,
        "sampling_errors": [],
        "violation": None,
        "termination": {"signals": [], "errors": []},
        "cleanup": {"process_group_gone": True},
        "stdout": file_identity(stdout_path),
        "stderr": file_identity(stderr_path),
    }


def fixture_raw(
    root: Path,
    source_root: Path,
    source: dict[str, Any],
    gpu_uuid: str,
) -> tuple[Path, Path, Path]:
    raw = root / "raw"
    (raw / "catalog").mkdir(parents=True)
    (raw / "catalog-export").mkdir()
    (raw / "build").mkdir()
    (raw / "binary").mkdir()
    (raw / "source-snapshot").mkdir()
    (raw / "bootstrap-native-operator-set").mkdir()
    provider_value, capability_value = fixture_catalogs()
    provider_path = raw / "catalog/provider-catalog.json"
    capability_path = raw / "catalog/capability-catalog.json"
    write_bytes_create_new(provider_path, rust_pretty_json_bytes(provider_value))
    write_bytes_create_new(capability_path, rust_pretty_json_bytes(capability_value))
    provider = validate_provider_catalog_bytes(provider_path.read_bytes())
    capability = validate_capability_catalog_bytes(capability_path.read_bytes(), cuda_ordinal=0)
    binary = raw / "binary/runtime_vnext_cuda_catalog_input"
    write_bytes_create_new(binary, b"fixture exporter\n")
    binary.chmod(0o755)
    command = [
        str(binary.resolve()),
        "0",
        "native-adaptive",
        str(provider_path.resolve()),
        str(capability_path.resolve()),
    ]
    collector_source = (
        source_root / "scripts/release/runtime_vnext_g03_live_catalog_collect.py"
    )
    collector_snapshot = raw / "source-snapshot" / collector_source.name
    shutil.copy2(collector_source, collector_snapshot)
    bootstrap_lock = (
        raw / "bootstrap-native-operator-set/native-operator-set.lock.json"
    )
    created_lock = create_selftest_native_operator_set(
        bootstrap_lock.parent, REQUIRED_CUDA_NATIVE_OPERATORS
    )
    require(created_lock == bootstrap_lock, "native operator fixture lock path drifted")
    validated_bootstrap = validate_native_operator_set(
        bootstrap_lock, REQUIRED_CUDA_NATIVE_OPERATORS
    )
    bootstrap = {
        "role": "build bootstrap only; G07B must rebuild artifacts against the exported live catalog",
        "lock": file_identity(bootstrap_lock, relative_to=raw),
        **native_operator_set_public_identity(validated_bootstrap),
    }
    target_dir = root / "target"
    native_cache = root / "native-cache"
    native_import = root / "native-import"
    target_dir.mkdir()
    native_cache.mkdir()
    native_import.mkdir()
    build_summary_path = raw / "build/cuda-build-summary.receipt.json"
    binary_rows = ",".join(
        f"{row['operator']}={row['sha256']}"
        for row in bootstrap["binary_sha256_by_operator"]
    )
    unit_names = ",".join(unit[1] for unit in REQUIRED_CUDA_NATIVE_BUILD_UNITS)
    native_signature = (
        f"lock={bootstrap_lock.resolve()}:"
        f"lock_sha256={bootstrap['lock']['sha256']}:"
        f"catalog={bootstrap['g03_catalog_sha256']}:"
        f"operators={len(REQUIRED_CUDA_NATIVE_BUILD_UNITS)}:"
        f"operator_binaries={binary_rows}:build_units={unit_names}"
    )
    build_rows = [
        {
            "artifact": "native_operator_artifact_set",
            "status": "linked",
            "reason": "manifest-v3-artifact-set-v5-validated",
            "elapsed_ms": 0,
            "inputs_hash": cuda_build_inputs_hash(native_signature),
        },
        {
            "artifact": "core-ptx:kernels/fixture.cu",
            "status": "cache_hit",
            "reason": "signature-match",
            "elapsed_ms": 0,
            "inputs_hash": "sha256:" + "d" * 64,
        },
    ]
    for summary_artifact, _unit_name, operator in REQUIRED_CUDA_NATIVE_BUILD_UNITS:
        build_rows.append(
            {
                "artifact": summary_artifact,
                "status": "artifact",
                "reason": "native-operator-artifact-set",
                "elapsed_ms": 0,
                "inputs_hash": cuda_build_inputs_hash(operator),
            }
        )
    write_json_create_new(
        build_summary_path,
        {
            "schema_version": 1,
            "artifact_type": "ferrum_cuda_build_summary_receipt",
            "rows": build_rows,
        },
    )
    build_command = [
        "/usr/bin/env",
        "PATH=/fixture/bin",
        "RUSTC=/fixture/bin/rustc",
        f"CARGO_TARGET_DIR={target_dir.resolve()}",
        "CARGO_BUILD_JOBS=4",
        "CUDA_COMPUTE_CAP=89",
        "FERRUM_NVCC_THREADS=4",
        f"FERRUM_CUDA_NATIVE_BUILD_CACHE={native_cache.resolve()}",
        f"FERRUM_CUDA_NATIVE_IMPORT_DIRS={native_import.resolve()}",
        f"FERRUM_NATIVE_OPERATOR_SET_LOCK={bootstrap_lock.resolve()}",
        "FERRUM_CUDA_NATIVE_SOURCE_POLICY=cache-only",
        f"FERRUM_CUDA_BUILD_SUMMARY_RECEIPT={build_summary_path.resolve()}",
        "/fixture/bin/cargo",
        "build",
        "--profile",
        "cuda-correctness",
        "--locked",
        "--jobs",
        "4",
        "-p",
        "ferrum-kernels",
        "--example",
        "runtime_vnext_cuda_catalog_input",
        "--features",
        "cuda,vllm-moe-marlin,vllm-paged-attn-v2",
    ]
    build_stdout = raw / "build/stdout.log"
    build_stderr = raw / "build/stderr.log"
    write_bytes_create_new(build_stdout, b"fixture build pass\n")
    write_bytes_create_new(build_stderr, b"")
    write_json_create_new(
        raw / "build/plan.json",
        {
            "schema_version": 1,
            "step_id": "build",
            "command": build_command,
            "cwd": str(source_root),
            "expected_duration_seconds": 10,
            "hard_deadline_seconds": 30,
            "progress_signal": "fixture build progress",
            "worker_limits": {
                "max_processes": 16,
                "max_group_threads": 192,
                "max_per_process_threads": 64,
            },
            "started_at": "2026-08-05T00:00:00.000Z",
        },
    )
    build_receipt_path = raw / "build/bounded.receipt.json"
    write_json_create_new(
        build_receipt_path,
        fixture_receipt(
            raw,
            source_root,
            build_command,
            build_stdout,
            build_stderr,
            max_processes=16,
            max_group_threads=192,
            max_per_process_threads=64,
        ),
    )
    stdout_path = raw / "catalog-export/stdout.log"
    stderr_path = raw / "catalog-export/stderr.log"
    write_bytes_create_new(
        stdout_path,
        (
            f"{READY_PREFIX} provider={provider_path.resolve()} capability={capability_path.resolve()} "
            f"provider_count=1 capability_fingerprint={capability['runtime_fingerprint']}\n"
        ).encode(),
    )
    write_bytes_create_new(stderr_path, b"")
    plan = {
        "schema_version": 1,
        "step_id": "catalog-export",
        "command": command,
        "cwd": str(source_root),
        "expected_duration_seconds": 5,
        "hard_deadline_seconds": 30,
        "progress_signal": "fixture progress",
        "worker_limits": {
            "max_processes": 2,
            "max_group_threads": 32,
            "max_per_process_threads": 32,
        },
        "started_at": "2026-08-05T00:00:00.000Z",
    }
    plan_path = raw / "catalog-export/plan.json"
    write_json_create_new(plan_path, plan)
    receipt_path = raw / "catalog-export/bounded.receipt.json"
    write_json_create_new(
        receipt_path,
        fixture_receipt(raw, source_root, command, stdout_path, stderr_path),
    )
    hardware = {
        "policy": "cuda-g0-1x-rtx4090",
        "gpu_count": 1,
        "gpu": f"0, NVIDIA GeForce RTX 4090, {gpu_uuid}, 24564 MiB, 555.00",
        "nvidia_smi": "fixture nvidia-smi",
        "nvcc": "fixture nvcc",
        "cargo": "fixture cargo",
        "rustc": "fixture rustc",
        "tools": {
            name: {"path": f"/fixture/{name}", "sha256": digit * 64, "size_bytes": 1}
            for name, digit in (
                ("nvidia_smi", "7"),
                ("nvcc", "8"),
                ("cargo", "9"),
                ("rustc", "a"),
            )
        },
    }
    write_json_create_new(raw / "source.json", source)
    write_json_create_new(raw / "hardware.json", hardware)
    provider_summary, capability_summary = raw_summary_from_catalogs(provider, capability)
    build_summary = validate_build_summary(
        build_summary_path,
        source_root=source_root,
        build_lock_path=bootstrap_lock,
        bootstrap=bootstrap,
    )
    manifest = {
        "schema_version": 1,
        "artifact_type": RAW_ARTIFACT_TYPE,
        "status": "ready",
        "created_at": "2026-08-05T00:00:02.000Z",
        "source": source,
        "hardware": hardware,
        "collector": {
            "source_path": "scripts/release/runtime_vnext_g03_live_catalog_collect.py",
            **file_identity(collector_snapshot, relative_to=raw),
        },
        "scope": {
            "backend": "cuda",
            "gpu_count": 1,
            "gpu_model": "RTX 4090",
            "cuda_ordinal": 0,
            "attention_policy": "native-adaptive",
            "cargo_profile": "cuda-correctness",
            "cargo_jobs": 4,
            "features": ["cuda", "vllm-moe-marlin", "vllm-paged-attn-v2"],
        },
        "bootstrap_native_operator_set": bootstrap,
        "build": {
            "command": build_command,
            "portable_command": portable_command(
                build_command,
                source_root=source_root,
                raw_root=raw,
                target_dir=target_dir,
                native_build_cache=native_cache,
                native_import_dirs=[native_import],
            ),
            "receipt": file_identity(build_receipt_path, relative_to=raw),
            "summary": build_summary,
            "summary_receipt": file_identity(
                build_summary_path, relative_to=raw
            ),
            "native_build_cache": str(native_cache.resolve()),
            "native_import_dirs": [str(native_import.resolve())],
        },
        "export": {
            "command": command,
            "portable_command": portable_command(
                command,
                source_root=source_root,
                raw_root=raw,
                target_dir=target_dir,
                native_build_cache=native_cache,
                native_import_dirs=[native_import],
            ),
            "binary": file_identity(binary, relative_to=raw),
            "receipt": file_identity(receipt_path, relative_to=raw),
            "receipt_status": "pass",
            "readiness": {
                "line": (
                    f"{READY_PREFIX} provider={provider_path.resolve()} "
                    f"capability={capability_path.resolve()} provider_count=1 "
                    f"capability_fingerprint={capability['runtime_fingerprint']}"
                ),
                "provider_count": 1,
                "capability_fingerprint": capability["runtime_fingerprint"],
            },
            "provider_catalog": {"path": "catalog/provider-catalog.json", **provider_summary},
            "capability_catalog": {"path": "catalog/capability-catalog.json", **capability_summary},
        },
        "does_not_prove": RAW_DOES_NOT_PROVE,
    }
    manifest["artifacts"] = checkpoint_artifact_index(raw)
    manifest["artifact_count"] = len(manifest["artifacts"])
    write_json_create_new(raw / "raw.manifest.json", manifest)
    return provider_path, capability_path, receipt_path


def refresh_tampered_hashes(output: Path, raw_root: Path) -> None:
    raw_manifest_path = raw_root / "raw.manifest.json"
    raw_manifest, _ = read_json(raw_manifest_path, "tampered raw manifest")
    raw_value = require_dict(raw_manifest, "tampered raw manifest")
    provider_path = raw_root / "catalog/provider-catalog.json"
    provider = validate_provider_catalog_bytes(provider_path.read_bytes())
    provider_summary, _ = raw_summary_from_catalogs(
        provider,
        validate_capability_catalog_bytes(
            (raw_root / "catalog/capability-catalog.json").read_bytes(), cuda_ordinal=0
        ),
    )
    raw_value["export"]["provider_catalog"] = {
        "path": "catalog/provider-catalog.json",
        **provider_summary,
    }
    for row in raw_value["artifacts"]:
        if row["path"] == "catalog/provider-catalog.json":
            row.update(file_identity(provider_path, relative_to=raw_root))
    raw_manifest_path.unlink()
    write_json_create_new(raw_manifest_path, raw_value)

    checkpoint_path = output / "manifest.json"
    checkpoint, _ = read_json(checkpoint_path, "tampered checkpoint")
    checkpoint_value = require_dict(checkpoint, "tampered checkpoint")
    copied = output / "provider-catalog.json"
    copied.write_bytes(provider_path.read_bytes())
    copied_identity = file_identity(copied, relative_to=output)
    checkpoint_value["artifacts"]["provider_catalog"] = copied_identity
    for row in checkpoint_value["artifact_index"]:
        if row["path"] == "provider-catalog.json":
            row.update(copied_identity)
    checkpoint_value["artifact_index_sha256"] = canonical_index_sha256(
        checkpoint_value["artifact_index"]
    )
    checkpoint_value["dependencies"]["raw_collection"]["manifest"] = external_ref(
        raw_manifest_path
    )
    checkpoint_path.unlink()
    write_json_create_new(checkpoint_path, checkpoint_value)


def self_test() -> str:
    with tempfile.TemporaryDirectory(prefix="ferrum-g03-live-checkpoint-") as temporary:
        root = Path(temporary).resolve()
        source_root, source = fixture_source(root)
        gpu_uuid = "GPU-11111111-2222-3333-4444-555555555555"
        s1_manifest = fixture_s1(root, source, gpu_uuid)
        provider, capability, receipt = fixture_raw(root, source_root, source, gpu_uuid)
        output = root / "checkpoint"
        output.mkdir()
        args = argparse.Namespace(
            source_root=source_root,
            s1_manifest=s1_manifest,
            provider_catalog=provider,
            capability_catalog=capability,
            catalog_export_receipt=receipt,
            out=output,
        )
        pass_line = execute(args)
        require(pass_line == f"{PASS_PREFIX}: {output}", "self-test PASS line mismatch")
        verify_checkpoint_manifest(
            output / "manifest.json", source_root=source_root, verify_checkout=True
        )

        extra = output / "extra.json"
        extra.write_text("{}\n", encoding="utf-8")
        expect_reject(
            lambda: verify_checkpoint_manifest(
                output / "manifest.json", source_root=source_root, verify_checkout=True
            ),
            "extra artifact",
        )
        extra.unlink()
        symlink = output / "linked.json"
        symlink.symlink_to(output / "provider-catalog.json")
        expect_reject(
            lambda: verify_checkpoint_manifest(
                output / "manifest.json", source_root=source_root, verify_checkout=True
            ),
            "symlink artifact",
        )
        symlink.unlink()

        manifest_path = output / "manifest.json"
        original_manifest = manifest_path.read_bytes()
        manifest, _ = read_json(manifest_path, "path traversal fixture")
        manifest["artifacts"]["provider_catalog"]["path"] = "../provider-catalog.json"
        manifest_path.unlink()
        write_json_create_new(manifest_path, manifest)
        expect_reject(
            lambda: verify_checkpoint_manifest(
                manifest_path, source_root=source_root, verify_checkout=True
            ),
            "path traversal",
        )
        manifest_path.unlink()
        write_bytes_create_new(manifest_path, original_manifest)

        provider_value, _ = read_json(provider, "semantic tamper provider")
        provider_value["providers"][0]["provider_version"] = {"major": 2, "minor": 0}
        provider.unlink()
        write_bytes_create_new(provider, rust_pretty_json_bytes(provider_value))
        refresh_tampered_hashes(output, receipt.parent.parent)
        expect_reject(
            lambda: verify_checkpoint_manifest(
                output / "manifest.json", source_root=source_root, verify_checkout=True
            ),
            "semantic provider tamper with synchronized outer hashes",
        )
        return pass_line


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-root", type=Path, default=REPO_ROOT)
    result.add_argument("--s1-manifest", type=Path)
    result.add_argument("--provider-catalog", type=Path)
    result.add_argument("--capability-catalog", type=Path)
    result.add_argument("--catalog-export-receipt", type=Path)
    result.add_argument("--out", type=Path)
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.self_test:
            require(
                all(
                    value is None
                    for value in (
                        args.s1_manifest,
                        args.provider_catalog,
                        args.capability_catalog,
                        args.catalog_export_receipt,
                        args.out,
                    )
                ),
                "--self-test cannot be combined with artifact arguments",
            )
            print(self_test())
            return 0
        missing = [
            flag
            for flag, value in (
                ("--s1-manifest", args.s1_manifest),
                ("--provider-catalog", args.provider_catalog),
                ("--capability-catalog", args.capability_catalog),
                ("--catalog-export-receipt", args.catalog_export_receipt),
                ("--out", args.out),
            )
            if value is None
        ]
        require(not missing, "missing required arguments: " + ", ".join(missing))
        print(execute(args))
        return 0
    except (CheckpointError, OSError, ValueError) as error:
        output = args.out.expanduser().resolve() if args.out is not None else Path("<unset>")
        print(f"FERRUM RUNTIME VNEXT G03 LIVE CATALOG FAIL: {output}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
