#!/usr/bin/env python3
"""Validate CUDA vNext operation determinism evidence.

The collector is the production VNextModelExecutor path. This validator owns
only the durable evidence contract: live-plan coverage, exact raw digests,
replay attribution, shape/state partitions, and freshness.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:
    import runtime_vnext_hardware_probe as hardware_probe
except ModuleNotFoundError:
    from scripts.release import runtime_vnext_hardware_probe as hardware_probe


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_CATALOG_PATH = REPO_ROOT / "scripts/release/configs/runtime_vnext_models.json"
HARDWARE_PROBE_PATH = REPO_ROOT / "scripts/release/runtime_vnext_hardware_probe.py"
PASS_PREFIX = "FERRUM RUNTIME VNEXT CUDA DETERMINISM PASS"
ARTIFACT_TYPE = "runtime_vnext_cuda_determinism_evidence"
VALIDATOR_ARTIFACT_TYPE = "runtime_vnext_cuda_determinism_validation"
PRIMARY_MODEL_LANES = {
    "m1-qwen35-4b": "M1-CUDA",
    "m2-qwen35-35b-a3b": "M2-CUDA",
    "m3-qwen3-30b-a3b": "M3-CUDA",
}
PRIMARY_MODELS = set(PRIMARY_MODEL_LANES)
REQUIRED_PARTITIONS = {
    ("prefill", "single_token"),
    ("prefill", "multi_token"),
    ("prefill", "chunk_boundary"),
    ("decode", "c1"),
    ("decode", "multi_participant"),
    ("decode", "c32"),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
PORTABLE_ID_RE = re.compile(r"^[A-Za-z0-9._:/-]+$")
MAX_CASES = 16_384
MAX_EXECUTIONS_PER_CASE = 64
MAX_WITNESSES_PER_EXECUTION = 131_072
MAX_COMPARISONS_PER_KIND = 32
MAX_ROOT_JSON_BYTES = 4 * 1024 * 1024
MAX_RECEIPT_JSON_BYTES = 4 * 1024 * 1024
MAX_MODEL_LOCK_JSON_BYTES = 64 * 1024 * 1024
MAX_DENOMINATOR_JSON_BYTES = 128 * 1024 * 1024
MAX_CASE_JSON_BYTES = 32 * 1024 * 1024
MAX_LOG_BYTES = 128 * 1024 * 1024
MAX_BINARY_BYTES = 1024 * 1024 * 1024

ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "backend",
        "source",
        "hardware",
        "models_lock",
        "denominator",
        "models",
        "runner",
        "cases",
    }
)
SOURCE_FIELDS = frozenset(
    {
        "git_sha",
        "git_tree_sha",
        "dirty_status",
        "build_command",
        "binary_path",
        "binary",
    }
)
HARDWARE_FIELDS = frozenset({"probe", "fingerprint"})
FILE_REF_FIELDS = frozenset({"path", "sha256", "size_bytes"})
DENOMINATOR_REF_FIELDS = FILE_REF_FIELDS | {"fingerprint"}
MODEL_FIELDS = frozenset(
    {
        "model_key",
        "source_model_id",
        "revision",
        "files",
        "config_sha256",
        "external_metadata_id",
        "resolved_plan_fingerprint",
        "plan_hash",
    }
)
MODEL_FILE_FIELDS = frozenset({"path", "sha256", "size_bytes"})
RUNNER_FIELDS = frozenset(
    {
        "command",
        "started_at",
        "finished_at",
        "exit_code",
        "receipt",
        "stdout",
        "stderr",
    }
)
BOUNDED_RECEIPT_FIELDS = frozenset(
    {
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
)
BOUNDED_LIMIT_FIELDS = frozenset(
    {
        "wall_timeout_seconds",
        "max_processes",
        "max_group_threads",
        "max_per_process_threads",
        "sample_interval_seconds",
        "max_sampling_errors",
        "term_grace_seconds",
    }
)
BOUNDED_PEAK_FIELDS = frozenset(
    {
        "processes",
        "group_threads",
        "per_process_threads",
        "per_process_threads_pid",
    }
)
BOUNDED_OUTPUT_FIELDS = frozenset({"path", "sha256", "size_bytes"})
COVERAGE_FIELDS = frozenset(
    {
        "schema_version",
        "device_id",
        "device_runtime_implementation_fingerprint",
        "capability_catalog_fingerprint",
        "models",
        "provider_requirements",
    }
)
COVERAGE_MODEL_FIELDS = frozenset(
    {
        "model_key",
        "external_metadata_id",
        "resolved_plan_fingerprint",
        "plan_hash",
        "node_ids",
    }
)
PROVIDER_REQUIREMENT_FIELDS = frozenset(
    {
        "operation_id",
        "operation_version",
        "operation_fingerprint",
        "provider_id",
        "provider_version",
        "provider_implementation_fingerprint",
        "provider_execution_contract_fingerprint",
        "replay_equivalence",
        "required_comparisons",
        "model_selections",
    }
)
MODEL_SELECTION_FIELDS = frozenset(
    {
        "model_key",
        "resolved_plan_fingerprint",
        "plan_hash",
        "node_ids",
    }
)
DENOMINATOR_FIELDS = frozenset(
    {
        "schema_version",
        "coverage",
        "provider_evidence",
    }
)
PROVIDER_EVIDENCE_FIELDS = frozenset(
    {
        "model_key",
        "resolved_plan_fingerprint",
        "plan_hash",
        "operation_id",
        "operation_fingerprint",
        "provider_id",
        "provider_implementation_fingerprint",
        "provider_execution_contract_fingerprint",
        "replay_equivalence",
        "required_comparisons",
        "node_ids",
        "witness_plan_fingerprint",
        "witness_plan",
    }
)
WITNESS_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "plan_hash",
        "node_ids",
        "replay_provider_requirements",
        "initializations",
        "witnesses",
    }
)
WITNESS_PROVIDER_REQUIREMENT_FIELDS = frozenset(
    {
        "provider_id",
        "provider_implementation_fingerprint",
        "provider_execution_contract_fingerprint",
        "node_ids",
    }
)
INITIALIZATION_SPEC_FIELDS = frozenset({"kind", "location", "consumer_node_ids"})
INITIALIZATION_KIND_EXTERNAL_FIELDS = frozenset({"kind", "value_id"})
INITIALIZATION_KIND_STATE_FIELDS = frozenset(
    {"kind", "state_id", "state_value_id", "lifetime", "access"}
)
WITNESS_SPEC_FIELDS = frozenset(
    {
        "provider_id",
        "provider_implementation_fingerprint",
        "provider_execution_contract_fingerprint",
        "kind",
        "location",
    }
)
WITNESS_KIND_OUTPUT_FIELDS = frozenset({"kind", "value_id", "output_ordinal"})
WITNESS_KIND_STATE_FIELDS = frozenset(
    {"kind", "state_id", "state_value_id", "lifetime", "access"}
)
VALUE_LOCATION_FIELDS = frozenset(
    {
        "node_id",
        "value_id",
        "role",
        "ordinal",
        "usage",
        "storage_component_ordinal",
        "storage_component_id",
        "resource_id",
        "logical_offset_bytes",
        "declared_length_bytes",
        "element_type",
        "extent",
    }
)
VALUE_EXTENT_FIXED_FIELDS = frozenset({"kind"})
VALUE_EXTENT_TOKEN_FIELDS = frozenset({"kind", "bytes_per_token", "maximum_tokens"})
VALUE_EXTENT_STATE_FIELDS = frozenset(
    {"kind", "bytes_per_token", "maximum_tokens", "maximum_storage_bytes"}
)
HARDWARE_PROBE_FIELDS = frozenset(
    {
        "schema_version",
        "source_git_sha",
        "source_tree_sha",
        "dirty_status",
        "collector",
        "hardware_id",
        "normalized",
        "fingerprint",
        "commands",
    }
)
HARDWARE_COMMAND_FIELDS = frozenset(
    {
        "kind",
        "argv",
        "returncode",
        "started_at",
        "finished_at",
        "duration_sec",
        "stdout",
        "stdout_sha256",
        "stderr",
        "stderr_sha256",
    }
)
CASE_FIELDS = frozenset(
    {
        "schema_version",
        "case_id",
        "denominator_fingerprint",
        "binary_sha256",
        "device_runtime_implementation_fingerprint",
        "device_fingerprint",
        "model_key",
        "resolved_plan_fingerprint",
        "plan_hash",
        "phase",
        "token_shape",
        "dtype",
        "quantization",
        "initialization",
        "coverage_targets",
        "executions",
        "comparisons",
        "first_mismatch",
    }
)
TOKEN_SHAPE_FIELDS = frozenset(
    {
        "partition",
        "participant_count",
        "immediate_tokens",
        "source_start_tokens",
        "source_end_tokens",
    }
)
INITIALIZATION_FIELDS = frozenset(
    {
        "input_sha256",
        "rng_sha256",
        "initial_state_kind",
        "initial_state_sha256",
        "workspace_poison",
    }
)
TARGET_FIELDS = frozenset(
    {
        "operation_id",
        "operation_version",
        "operation_fingerprint",
        "provider_id",
        "provider_version",
        "provider_implementation_fingerprint",
        "provider_execution_contract_fingerprint",
        "replay_equivalence",
        "witness_plan_fingerprint",
        "node_ids",
    }
)
EXECUTION_FIELDS = frozenset(
    {
        "execution_id",
        "mode",
        "restore_sha256",
        "submission_fingerprint",
        "receipt_fingerprint",
        "attribution",
        "witnesses",
    }
)
ATTRIBUTION_FIELDS = frozenset(
    {
        "batch_identity_fingerprint",
        "submission_fingerprint",
        "reusable_executable_fingerprint",
        "commands",
    }
)
COMMAND_FIELDS = frozenset(
    {
        "command_index",
        "node_id",
        "command_phase",
        "native_op_id",
        "execution_path",
        "batching_form",
        "participant_count",
        "token_count",
        "compute_dispatch_count",
        "transfer_command_count",
        "reusable_graph_node_count",
    }
)
WITNESS_FIELDS = frozenset(
    {
        "kind",
        "semantic_id",
        "node_id",
        "resource_id",
        "access",
        "participant_index",
        "logical_offset_bytes",
        "length_bytes",
        "element_type",
        "raw_sha256",
    }
)
COMPARISON_FIELDS = frozenset(
    {
        "kind",
        "ordinal",
        "left_execution_id",
        "right_execution_id",
        "relation",
        "first_mismatch",
    }
)


class DeterminismArtifactError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DeterminismArtifactError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def reject_constant(value: str) -> None:
    raise DeterminismArtifactError(f"non-finite JSON constant is forbidden: {value}")


def read_json(path: Path, *, max_bytes: int = MAX_DENOMINATOR_JSON_BYTES) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        require(not path.is_symlink() and path.is_file(), f"{path} must be a real JSON file")
        require(
            0 < metadata.st_size <= max_bytes,
            f"{path} exceeds its JSON byte bound {max_bytes}",
        )
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except DeterminismArtifactError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeterminismArtifactError(f"invalid JSON {path}: {error}") from error
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_structural_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def exact_object(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    require(not fields - actual, f"{label} is missing fields: {sorted(fields - actual)}")
    require(not actual - fields, f"{label} has unknown fields: {sorted(actual - fields)}")
    return value


def text(value: Any, label: str, *, portable: bool = False) -> str:
    require(
        isinstance(value, str) and value == value.strip() and bool(value),
        f"{label} must be a non-empty trimmed string",
    )
    if portable:
        require(
            len(value) <= 256 and PORTABLE_ID_RE.fullmatch(value) is not None,
            f"{label} must be a portable identity",
        )
    return value


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return value


def number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) >= minimum,
        f"{label} must be a finite number >= {minimum}",
    )
    result = float(value)
    require(result < float("inf"), f"{label} must be finite")
    return result


def sha256_text(value: Any, label: str) -> str:
    digest = text(value, label)
    require(SHA256_RE.fullmatch(digest) is not None, f"{label} must be a lowercase SHA256")
    return digest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structural_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safe_artifact_file(root: Path, relative: Any, label: str) -> Path:
    relative_text = text(relative, f"{label}.path")
    relative_path = Path(relative_text)
    require(
        not relative_path.is_absolute()
        and relative_path.parts
        and ".." not in relative_path.parts
        and relative_path.as_posix() == relative_text,
        f"{label}.path must be a canonical relative POSIX path",
    )
    current = root
    for index, part in enumerate(relative_path.parts):
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise DeterminismArtifactError(f"{label}.path is missing: {current}: {error}") from error
        require(not current.is_symlink(), f"{label}.path cannot traverse a symlink")
        if index + 1 < len(relative_path.parts):
            require(current.is_dir(), f"{label}.path parent must be a directory")
    require(current.is_file(), f"{label}.path must be a real file")
    return current


def validate_file_ref(
    root: Path,
    value: Any,
    label: str,
    *,
    max_size_bytes: int = MAX_DENOMINATOR_JSON_BYTES,
) -> tuple[dict[str, Any], Path]:
    ref = exact_object(value, FILE_REF_FIELDS, label)
    path = safe_artifact_file(root, ref["path"], label)
    size = integer(ref["size_bytes"], f"{label}.size_bytes", minimum=1)
    require(size <= max_size_bytes, f"{label}.size_bytes exceeds {max_size_bytes}")
    require(path.stat().st_size == size, f"{label}.size_bytes differs from the file")
    digest = sha256_text(ref["sha256"], f"{label}.sha256")
    require(file_sha256(path) == digest, f"{label}.sha256 differs from the file")
    return ref, path


def validate_version(value: Any, label: str) -> tuple[int, int]:
    version = exact_object(value, frozenset({"major", "minor"}), label)
    major = integer(version["major"], f"{label}.major", minimum=1)
    minor = integer(version["minor"], f"{label}.minor")
    return major, minor


def validate_string_list(
    value: Any,
    label: str,
    *,
    portable: bool = False,
    nonempty: bool = True,
) -> list[str]:
    require(isinstance(value, list), f"{label} must be a list")
    if nonempty:
        require(bool(value), f"{label} must not be empty")
    result = [text(item, f"{label}[{index}]", portable=portable) for index, item in enumerate(value)]
    require(result == sorted(set(result)), f"{label} must be sorted and unique")
    return result


def validate_timestamp(value: Any, label: str) -> str:
    timestamp = text(value, label)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise DeterminismArtifactError(f"{label} is not ISO-8601: {error}") from error
    require(parsed.tzinfo is not None, f"{label} must include a timezone")
    return timestamp


def validate_bounded_receipt(
    receipt: dict[str, Any],
    *,
    command: list[str],
    started_at: str,
    finished_at: str,
    exit_code: int,
    stdout_ref: dict[str, Any],
    stderr_ref: dict[str, Any],
) -> None:
    receipt = exact_object(receipt, BOUNDED_RECEIPT_FIELDS, "runner.receipt")
    require(
        receipt["schema"] == "ferrum.bounded-command-receipt.v1",
        "runner.receipt schema is invalid",
    )
    require(receipt["command"] == command, "runner.receipt command differs from runner.command")
    text(receipt["cwd"], "runner.receipt.cwd")
    pid = integer(receipt["pid"], "runner.receipt.pid", minimum=1)
    require(receipt["pgid"] == pid, "runner.receipt must own its process group")

    limits = exact_object(receipt["limits"], BOUNDED_LIMIT_FIELDS, "runner.receipt.limits")
    wall_timeout = number(
        limits["wall_timeout_seconds"],
        "runner.receipt.limits.wall_timeout_seconds",
        minimum=1.0,
    )
    require(wall_timeout <= 6 * 60 * 60, "runner receipt wall timeout exceeds six hours")
    max_processes = integer(
        limits["max_processes"], "runner.receipt.limits.max_processes", minimum=1
    )
    max_group_threads = integer(
        limits["max_group_threads"],
        "runner.receipt.limits.max_group_threads",
        minimum=1,
    )
    max_per_process_threads = integer(
        limits["max_per_process_threads"],
        "runner.receipt.limits.max_per_process_threads",
        minimum=1,
    )
    require(max_processes <= 64, "runner receipt process bound exceeds 64")
    require(max_group_threads <= 1024, "runner receipt group-thread bound exceeds 1024")
    require(
        max_per_process_threads <= 512,
        "runner receipt per-process thread bound exceeds 512",
    )
    number(
        limits["sample_interval_seconds"],
        "runner.receipt.limits.sample_interval_seconds",
        minimum=0.001,
    )
    integer(
        limits["max_sampling_errors"],
        "runner.receipt.limits.max_sampling_errors",
        minimum=1,
    )
    number(
        limits["term_grace_seconds"],
        "runner.receipt.limits.term_grace_seconds",
    )

    peaks = exact_object(receipt["peaks"], BOUNDED_PEAK_FIELDS, "runner.receipt.peaks")
    peak_processes = integer(peaks["processes"], "runner.receipt.peaks.processes", minimum=1)
    peak_group_threads = integer(
        peaks["group_threads"], "runner.receipt.peaks.group_threads", minimum=1
    )
    peak_per_process_threads = integer(
        peaks["per_process_threads"],
        "runner.receipt.peaks.per_process_threads",
        minimum=1,
    )
    integer(
        peaks["per_process_threads_pid"],
        "runner.receipt.peaks.per_process_threads_pid",
        minimum=1,
    )
    require(
        peak_processes <= max_processes
        and peak_group_threads <= max_group_threads
        and peak_per_process_threads <= max_per_process_threads,
        "runner receipt peak exceeds its declared bound",
    )

    require(
        receipt["started_at"] == started_at and receipt["ended_at"] == finished_at,
        "runner receipt timestamps differ from runner identity",
    )
    number(receipt["duration_seconds"], "runner.receipt.duration_seconds", minimum=0.001)
    require(
        receipt["reason"] == "command_completed"
        and receipt["rc"] == exit_code == 0
        and receipt["status"] == "pass",
        "runner receipt did not record a successful command completion",
    )
    integer(
        receipt["successful_samples"],
        "runner.receipt.successful_samples",
        minimum=1,
    )
    require(
        receipt["sampling_error_count"] == 0
        and receipt["sampling_errors"] == [],
        "runner receipt contains sampling errors",
    )
    require(receipt["violation"] is None, "runner receipt contains a resource violation")
    termination = exact_object(
        receipt["termination"],
        frozenset({"signals", "errors"}),
        "runner.receipt.termination",
    )
    require(
        termination == {"signals": [], "errors": []},
        "runner receipt required termination",
    )
    cleanup = exact_object(
        receipt["cleanup"],
        frozenset({"process_group_gone"}),
        "runner.receipt.cleanup",
    )
    require(cleanup["process_group_gone"] is True, "runner receipt did not clean its process group")
    for name, expected in (("stdout", stdout_ref), ("stderr", stderr_ref)):
        output = exact_object(
            receipt[name], BOUNDED_OUTPUT_FIELDS, f"runner.receipt.{name}"
        )
        text(output["path"], f"runner.receipt.{name}.path")
        require(
            output["sha256"] == expected["sha256"]
            and output["size_bytes"] == expected["size_bytes"],
            f"runner receipt {name} identity differs from copied evidence",
        )


def validate_coverage(value: dict[str, Any]) -> dict[str, Any]:
    coverage = exact_object(value, COVERAGE_FIELDS, "coverage")
    require(validate_version(coverage["schema_version"], "coverage.schema_version") == (1, 0),
            "coverage.schema_version must be 1.0")
    text(coverage["device_id"], "coverage.device_id", portable=True)
    runtime_fingerprint = sha256_text(
        coverage["device_runtime_implementation_fingerprint"],
        "coverage.device_runtime_implementation_fingerprint",
    )
    catalog_fingerprint = sha256_text(
        coverage["capability_catalog_fingerprint"],
        "coverage.capability_catalog_fingerprint",
    )

    models_raw = coverage["models"]
    require(isinstance(models_raw, list), "coverage.models must be a list")
    require(len(models_raw) == len(PRIMARY_MODELS), "coverage must contain the three primary models")
    models: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(models_raw):
        model = exact_object(raw, COVERAGE_MODEL_FIELDS, f"coverage.models[{index}]")
        key = text(model["model_key"], f"coverage.models[{index}].model_key", portable=True)
        require(key in PRIMARY_MODELS and key not in models, "coverage model key is invalid or duplicated")
        text(model["external_metadata_id"], f"coverage.models[{index}].external_metadata_id", portable=True)
        sha256_text(
            model["resolved_plan_fingerprint"],
            f"coverage.models[{index}].resolved_plan_fingerprint",
        )
        sha256_text(model["plan_hash"], f"coverage.models[{index}].plan_hash")
        validate_string_list(
            model["node_ids"],
            f"coverage.models[{index}].node_ids",
            portable=True,
        )
        models[key] = model
    require(list(models) == sorted(models), "coverage.models must use canonical model-key order")

    requirements_raw = coverage["provider_requirements"]
    require(
        isinstance(requirements_raw, list) and bool(requirements_raw),
        "coverage.provider_requirements must be non-empty",
    )
    requirements: dict[tuple[str, str], dict[str, Any]] = {}
    previous_key: tuple[str, str] | None = None
    model_nodes: dict[str, set[str]] = {key: set() for key in PRIMARY_MODELS}
    for index, raw in enumerate(requirements_raw):
        label = f"coverage.provider_requirements[{index}]"
        requirement = exact_object(raw, PROVIDER_REQUIREMENT_FIELDS, label)
        operation_id = text(requirement["operation_id"], f"{label}.operation_id", portable=True)
        provider_id = text(requirement["provider_id"], f"{label}.provider_id", portable=True)
        key = (operation_id, provider_id)
        require(previous_key is None or previous_key < key, "coverage provider rows are not canonical")
        previous_key = key
        validate_version(requirement["operation_version"], f"{label}.operation_version")
        validate_version(requirement["provider_version"], f"{label}.provider_version")
        sha256_text(requirement["operation_fingerprint"], f"{label}.operation_fingerprint")
        sha256_text(
            requirement["provider_implementation_fingerprint"],
            f"{label}.provider_implementation_fingerprint",
        )
        sha256_text(
            requirement["provider_execution_contract_fingerprint"],
            f"{label}.provider_execution_contract_fingerprint",
        )
        replay = text(requirement["replay_equivalence"], f"{label}.replay_equivalence")
        require(
            replay in {"ineligible", "bitwise_eager_equivalent"},
            f"{label}.replay_equivalence is invalid",
        )
        expected_comparisons = (
            ["eager_eager"]
            if replay == "ineligible"
            else ["eager_eager", "replay_replay", "eager_replay"]
        )
        require(
            requirement["required_comparisons"] == expected_comparisons,
            f"{label}.required_comparisons differs from replay equivalence",
        )
        selections_raw = requirement["model_selections"]
        require(
            isinstance(selections_raw, list) and bool(selections_raw),
            f"{label}.model_selections cannot be empty",
        )
        selection_keys: list[str] = []
        for selection_index, selection_raw in enumerate(selections_raw):
            selection_label = f"{label}.model_selections[{selection_index}]"
            selection = exact_object(selection_raw, MODEL_SELECTION_FIELDS, selection_label)
            model_key = text(selection["model_key"], f"{selection_label}.model_key", portable=True)
            require(model_key in models, f"{selection_label}.model_key is absent from coverage models")
            require(
                selection["resolved_plan_fingerprint"] == models[model_key]["resolved_plan_fingerprint"]
                and selection["plan_hash"] == models[model_key]["plan_hash"],
                f"{selection_label} differs from the coverage model plan",
            )
            node_ids = validate_string_list(
                selection["node_ids"],
                f"{selection_label}.node_ids",
                portable=True,
            )
            require(
                set(node_ids) <= set(models[model_key]["node_ids"]),
                f"{selection_label}.node_ids escape the model plan",
            )
            overlap = model_nodes[model_key] & set(node_ids)
            require(not overlap, f"{selection_label}.node_ids duplicate another provider selection")
            model_nodes[model_key].update(node_ids)
            selection_keys.append(model_key)
        require(selection_keys == sorted(set(selection_keys)), f"{label}.model_selections are not canonical")
        requirements[key] = requirement
    for model_key, model in models.items():
        require(
            model_nodes[model_key] == set(model["node_ids"]),
            f"coverage provider selections do not partition model {model_key} nodes",
        )
    return {
        "device_id": coverage["device_id"],
        "runtime_fingerprint": runtime_fingerprint,
        "catalog_fingerprint": catalog_fingerprint,
        "models": models,
        "requirements": requirements,
    }


def validate_location(value: Any, label: str) -> dict[str, Any]:
    location = exact_object(value, VALUE_LOCATION_FIELDS, label)
    text(location["node_id"], f"{label}.node_id", portable=True)
    text(location["value_id"], f"{label}.value_id", portable=True)
    text(location["role"], f"{label}.role", portable=True)
    integer(location["ordinal"], f"{label}.ordinal")
    text(location["usage"], f"{label}.usage", portable=True)
    integer(location["storage_component_ordinal"], f"{label}.storage_component_ordinal")
    if location["storage_component_id"] is not None:
        text(
            location["storage_component_id"],
            f"{label}.storage_component_id",
            portable=True,
        )
    text(location["resource_id"], f"{label}.resource_id", portable=True)
    offset = integer(location["logical_offset_bytes"], f"{label}.logical_offset_bytes")
    declared = integer(
        location["declared_length_bytes"],
        f"{label}.declared_length_bytes",
        minimum=1,
    )
    require(offset + declared <= (1 << 64) - 1, f"{label} byte range overflows u64")
    text(location["element_type"], f"{label}.element_type", portable=True)
    extent = location["extent"]
    require(isinstance(extent, dict), f"{label}.extent must be an object")
    kind = text(extent.get("kind"), f"{label}.extent.kind")
    if kind == "fixed":
        exact_object(extent, VALUE_EXTENT_FIXED_FIELDS, f"{label}.extent")
        maximum_length = declared
    elif kind == "immediate_token_span":
        exact_object(extent, VALUE_EXTENT_TOKEN_FIELDS, f"{label}.extent")
        bytes_per_token = integer(
            extent["bytes_per_token"], f"{label}.extent.bytes_per_token", minimum=1
        )
        maximum_tokens = integer(
            extent["maximum_tokens"], f"{label}.extent.maximum_tokens", minimum=1
        )
        maximum_length = bytes_per_token * maximum_tokens
    elif kind == "active_token_prefix":
        exact_object(extent, VALUE_EXTENT_STATE_FIELDS, f"{label}.extent")
        bytes_per_token = integer(
            extent["bytes_per_token"], f"{label}.extent.bytes_per_token", minimum=1
        )
        maximum_tokens = integer(
            extent["maximum_tokens"], f"{label}.extent.maximum_tokens", minimum=1
        )
        maximum_storage = integer(
            extent["maximum_storage_bytes"],
            f"{label}.extent.maximum_storage_bytes",
            minimum=1,
        )
        require(
            bytes_per_token * maximum_tokens <= maximum_storage,
            f"{label}.extent token capacity exceeds maximum storage",
        )
        maximum_length = maximum_storage
    else:
        raise DeterminismArtifactError(f"{label}.extent.kind is invalid")
    require(
        maximum_length >= declared,
        f"{label}.extent is smaller than its declared location",
    )
    return location


def validate_witness_plan(
    value: Any,
    label: str,
    *,
    expected_plan_hash: str,
    expected_node_ids: list[str],
    expected_provider: dict[str, Any],
) -> dict[str, Any]:
    plan = exact_object(value, WITNESS_PLAN_FIELDS, label)
    require(
        validate_version(plan["schema_version"], f"{label}.schema_version") == (4, 0),
        f"{label}.schema_version must be 4.0",
    )
    require(plan["plan_hash"] == expected_plan_hash, f"{label}.plan_hash is stale")
    node_ids = validate_string_list(plan["node_ids"], f"{label}.node_ids", portable=True)
    require(node_ids == expected_node_ids, f"{label}.node_ids differ from provider scope")

    replay_requirements = plan["replay_provider_requirements"]
    require(
        isinstance(replay_requirements, list),
        f"{label}.replay_provider_requirements must be a list",
    )
    replay_nodes: set[str] = set()
    replay_provider_ids: list[str] = []
    for index, raw in enumerate(replay_requirements):
        item_label = f"{label}.replay_provider_requirements[{index}]"
        item = exact_object(raw, WITNESS_PROVIDER_REQUIREMENT_FIELDS, item_label)
        provider_id = text(item["provider_id"], f"{item_label}.provider_id", portable=True)
        sha256_text(
            item["provider_implementation_fingerprint"],
            f"{item_label}.provider_implementation_fingerprint",
        )
        sha256_text(
            item["provider_execution_contract_fingerprint"],
            f"{item_label}.provider_execution_contract_fingerprint",
        )
        nodes = set(
            validate_string_list(item["node_ids"], f"{item_label}.node_ids", portable=True)
        )
        require(nodes <= set(node_ids), f"{item_label}.node_ids escape provider scope")
        require(not replay_nodes & nodes, f"{item_label}.node_ids overlap another provider")
        replay_nodes.update(nodes)
        replay_provider_ids.append(provider_id)
    require(
        replay_provider_ids == sorted(set(replay_provider_ids)),
        f"{label}.replay_provider_requirements are not canonical",
    )
    if expected_provider["replay_equivalence"] == "bitwise_eager_equivalent":
        require(
            replay_provider_ids == [expected_provider["provider_id"]]
            and replay_nodes == set(node_ids),
            f"{label} lacks the exact replay provider denominator",
        )
        replay_requirement = replay_requirements[0]
        require(
            replay_requirement["provider_implementation_fingerprint"]
            == expected_provider["provider_implementation_fingerprint"]
            and replay_requirement["provider_execution_contract_fingerprint"]
            == expected_provider["provider_execution_contract_fingerprint"],
            f"{label} replay provider identity is stale",
        )
    else:
        require(
            replay_requirements == [],
            f"{label} eager-only provider cannot declare replay requirements",
        )

    initializations = plan["initializations"]
    require(
        isinstance(initializations, list) and len(initializations) <= 262_144,
        f"{label}.initializations cardinality is invalid",
    )
    initialization_rows: set[str] = set()
    for index, raw in enumerate(initializations):
        item_label = f"{label}.initializations[{index}]"
        item = exact_object(raw, INITIALIZATION_SPEC_FIELDS, item_label)
        kind = item["kind"]
        require(isinstance(kind, dict), f"{item_label}.kind must be an object")
        kind_name = text(kind.get("kind"), f"{item_label}.kind.kind")
        if kind_name == "external_input":
            exact_object(kind, INITIALIZATION_KIND_EXTERNAL_FIELDS, f"{item_label}.kind")
            text(kind["value_id"], f"{item_label}.kind.value_id", portable=True)
        elif kind_name == "state":
            exact_object(kind, INITIALIZATION_KIND_STATE_FIELDS, f"{item_label}.kind")
            for field in ("state_id", "state_value_id", "lifetime", "access"):
                text(kind[field], f"{item_label}.kind.{field}", portable=True)
        else:
            raise DeterminismArtifactError(f"{item_label}.kind.kind is invalid")
        location = validate_location(item["location"], f"{item_label}.location")
        require(
            location["node_id"] in node_ids,
            f"{item_label}.location node escapes provider scope",
        )
        consumers = validate_string_list(
            item["consumer_node_ids"],
            f"{item_label}.consumer_node_ids",
            portable=True,
        )
        require(
            set(consumers) <= set(node_ids),
            f"{item_label}.consumer_node_ids escape provider scope",
        )
        row = json.dumps(item, separators=(",", ":"), ensure_ascii=False)
        require(row not in initialization_rows, f"{item_label} duplicates an initialization")
        initialization_rows.add(row)

    witnesses = plan["witnesses"]
    require(
        isinstance(witnesses, list) and 0 < len(witnesses) <= 1_048_576,
        f"{label}.witnesses cardinality is invalid",
    )
    witness_rows: set[str] = set()
    output_nodes: set[str] = set()
    expected_runtime_witnesses: list[dict[str, Any]] = []
    for index, raw in enumerate(witnesses):
        item_label = f"{label}.witnesses[{index}]"
        item = exact_object(raw, WITNESS_SPEC_FIELDS, item_label)
        require(
            item["provider_id"] == expected_provider["provider_id"]
            and item["provider_implementation_fingerprint"]
            == expected_provider["provider_implementation_fingerprint"]
            and item["provider_execution_contract_fingerprint"]
            == expected_provider["provider_execution_contract_fingerprint"],
            f"{item_label} provider identity is stale",
        )
        kind = item["kind"]
        require(isinstance(kind, dict), f"{item_label}.kind must be an object")
        kind_name = text(kind.get("kind"), f"{item_label}.kind.kind")
        if kind_name == "output":
            exact_object(kind, WITNESS_KIND_OUTPUT_FIELDS, f"{item_label}.kind")
            semantic_id = text(
                kind["value_id"], f"{item_label}.kind.value_id", portable=True
            )
            integer(kind["output_ordinal"], f"{item_label}.kind.output_ordinal")
            runtime_kind = "declared_output"
            access = "write"
        elif kind_name == "state_effect":
            exact_object(kind, WITNESS_KIND_STATE_FIELDS, f"{item_label}.kind")
            semantic_id = text(
                kind["state_id"], f"{item_label}.kind.state_id", portable=True
            )
            for field in ("state_value_id", "lifetime", "access"):
                text(kind[field], f"{item_label}.kind.{field}", portable=True)
            runtime_kind = "state_effect"
            access = kind["access"]
        else:
            raise DeterminismArtifactError(f"{item_label}.kind.kind is invalid")
        location = validate_location(item["location"], f"{item_label}.location")
        require(
            location["node_id"] in node_ids,
            f"{item_label}.location node escapes provider scope",
        )
        if runtime_kind == "declared_output":
            output_nodes.add(location["node_id"])
        row = json.dumps(item, separators=(",", ":"), ensure_ascii=False)
        require(row not in witness_rows, f"{item_label} duplicates a witness")
        witness_rows.add(row)
        expected_runtime_witnesses.append(
            {
                "kind": runtime_kind,
                "semantic_id": semantic_id,
                "node_id": location["node_id"],
                "resource_id": location["resource_id"],
                "access": access,
                "element_type": location["element_type"],
                "location": location,
            }
        )
    require(output_nodes == set(node_ids), f"{label} lacks an output witness for every node")
    return {
        "witness_plan": plan,
        "expected_runtime_witnesses": expected_runtime_witnesses,
    }


def validate_denominator(value: dict[str, Any]) -> dict[str, Any]:
    denominator = exact_object(value, DENOMINATOR_FIELDS, "denominator")
    require(
        validate_version(denominator["schema_version"], "denominator.schema_version")
        == (1, 0),
        "denominator.schema_version must be 1.0",
    )
    coverage = validate_coverage(denominator["coverage"])
    evidence_rows = denominator["provider_evidence"]
    require(
        isinstance(evidence_rows, list)
        and 0 < len(evidence_rows) <= len(PRIMARY_MODELS) * 512,
        "denominator.provider_evidence cardinality is invalid",
    )
    scopes: dict[tuple[str, str, str], dict[str, Any]] = {}
    previous_key: tuple[str, str, str] | None = None
    for index, raw in enumerate(evidence_rows):
        label = f"denominator.provider_evidence[{index}]"
        evidence = exact_object(raw, PROVIDER_EVIDENCE_FIELDS, label)
        model_key = text(evidence["model_key"], f"{label}.model_key", portable=True)
        operation_id = text(
            evidence["operation_id"], f"{label}.operation_id", portable=True
        )
        provider_id = text(
            evidence["provider_id"], f"{label}.provider_id", portable=True
        )
        key = (model_key, operation_id, provider_id)
        require(previous_key is None or previous_key < key, f"{label} is not canonical")
        previous_key = key
        require(model_key in coverage["models"], f"{label}.model_key is unknown")
        requirement = coverage["requirements"].get((operation_id, provider_id))
        require(requirement is not None, f"{label} is absent from live coverage")
        selection = next(
            (
                item
                for item in requirement["model_selections"]
                if item["model_key"] == model_key
            ),
            None,
        )
        require(selection is not None, f"{label} has no live plan selection")
        model = coverage["models"][model_key]
        require(
            evidence["resolved_plan_fingerprint"]
            == selection["resolved_plan_fingerprint"]
            == model["resolved_plan_fingerprint"]
            and evidence["plan_hash"] == selection["plan_hash"] == model["plan_hash"]
            and evidence["operation_fingerprint"]
            == requirement["operation_fingerprint"]
            and evidence["provider_implementation_fingerprint"]
            == requirement["provider_implementation_fingerprint"]
            and evidence["provider_execution_contract_fingerprint"]
            == requirement["provider_execution_contract_fingerprint"]
            and evidence["replay_equivalence"] == requirement["replay_equivalence"]
            and evidence["required_comparisons"]
            == requirement["required_comparisons"]
            and evidence["node_ids"] == selection["node_ids"],
            f"{label} differs from the live catalog or resolved plan",
        )
        witness_fingerprint = sha256_text(
            evidence["witness_plan_fingerprint"],
            f"{label}.witness_plan_fingerprint",
        )
        require(
            structural_sha256(evidence["witness_plan"]) == witness_fingerprint,
            f"{label}.witness_plan_fingerprint is not derived from the typed witness plan",
        )
        witness = validate_witness_plan(
            evidence["witness_plan"],
            f"{label}.witness_plan",
            expected_plan_hash=evidence["plan_hash"],
            expected_node_ids=evidence["node_ids"],
            expected_provider=evidence,
        )
        scopes[key] = {
            **evidence,
            **witness,
        }
    expected = {
        (selection["model_key"], operation_id, provider_id)
        for (operation_id, provider_id), requirement in coverage["requirements"].items()
        for selection in requirement["model_selections"]
    }
    require(
        set(scopes) == expected,
        "denominator.provider_evidence does not equal the live coverage selection set",
    )
    return {
        **coverage,
        "scopes": scopes,
    }


def validate_model_catalog() -> tuple[dict[str, dict[str, Any]], str]:
    catalog = read_json(MODELS_CATALOG_PATH, max_bytes=MAX_ROOT_JSON_BYTES)
    require(
        catalog.get("schema_version") == 1,
        "checked-in runtime vNext model catalog schema is invalid",
    )
    rows = catalog.get("models")
    require(isinstance(rows, list), "checked-in runtime vNext model catalog has no models")
    by_lane = {
        row.get("id"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    result: dict[str, dict[str, Any]] = {}
    for model_key, lane_id in PRIMARY_MODEL_LANES.items():
        row = by_lane.get(lane_id)
        require(row is not None, f"checked-in model catalog lacks {lane_id}")
        revision = row.get("revision")
        require(
            row.get("backend") == "cuda"
            and isinstance(row.get("repo"), str)
            and isinstance(row.get("format"), str)
            and isinstance(revision, dict)
            and revision.get("status") == "pinned"
            and GIT_SHA_RE.fullmatch(str(revision.get("value"))) is not None,
            f"checked-in model catalog lane {lane_id} is not an immutable CUDA lane",
        )
        result[model_key] = row
    return result, file_sha256(MODELS_CATALOG_PATH)


def validate_models_lock(
    root: Path,
    value: Any,
) -> dict[str, dict[str, Any]]:
    _, path = validate_file_ref(
        root,
        value,
        "evidence.models_lock",
        max_size_bytes=MAX_MODEL_LOCK_JSON_BYTES,
    )
    lock = read_json(path, max_bytes=MAX_MODEL_LOCK_JSON_BYTES)
    require(lock.get("schema_version") == 1, "models.lock schema_version must be 1")
    catalog, catalog_sha = validate_model_catalog()
    require(
        lock.get("catalog_sha256") == catalog_sha,
        "models.lock is stale against the checked-in model catalog",
    )
    rows = lock.get("models")
    require(isinstance(rows, list), "models.lock.models must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        require(isinstance(raw, dict), f"models.lock.models[{index}] must be an object")
        key = raw.get("key")
        if key not in PRIMARY_MODELS:
            continue
        require(key not in indexed, f"models.lock duplicates primary model {key}")
        lanes = raw.get("lanes")
        require(
            isinstance(lanes, dict) and isinstance(lanes.get("cuda"), dict),
            f"models.lock model {key} lacks its CUDA lane",
        )
        lane = lanes["cuda"]
        expected = catalog[str(key)]
        require(
            lane.get("catalog_lane_id") == PRIMARY_MODEL_LANES[str(key)]
            and lane.get("repo") == expected["repo"]
            and lane.get("revision") == expected["revision"]["value"]
            and lane.get("format") == expected["format"],
            f"models.lock model {key} differs from the checked-in CUDA catalog lane",
        )
        files = lane.get("files")
        require(
            isinstance(files, list) and bool(files),
            f"models.lock model {key} files must be non-empty",
        )
        paths: list[str] = []
        normalized_files: list[dict[str, Any]] = []
        for file_index, file_raw in enumerate(files):
            label = f"models.lock.models[{index}].lanes.cuda.files[{file_index}]"
            require(isinstance(file_raw, dict), f"{label} must be an object")
            file_path = text(file_raw.get("path"), f"{label}.path")
            digest = sha256_text(file_raw.get("sha256"), f"{label}.sha256")
            size = integer(file_raw.get("size_bytes"), f"{label}.size_bytes", minimum=1)
            paths.append(file_path)
            normalized_files.append(
                {"path": file_path, "sha256": digest, "size_bytes": size}
            )
        require(paths == sorted(set(paths)), f"models.lock model {key} files are not canonical")
        config = next((item for item in normalized_files if item["path"] == "config.json"), None)
        require(config is not None, f"models.lock model {key} has no config.json lock")
        indexed[str(key)] = {
            "model_key": str(key),
            "source_model_id": lane["repo"],
            "revision": lane["revision"],
            "files": normalized_files,
            "config_sha256": config["sha256"],
        }
    require(
        set(indexed) == PRIMARY_MODELS,
        "models.lock must contain all three primary CUDA models",
    )
    return indexed


def validate_hardware(
    root: Path,
    value: Any,
    *,
    source: dict[str, Any],
) -> str:
    hardware = exact_object(value, HARDWARE_FIELDS, "evidence.hardware")
    _, probe_path = validate_file_ref(
        root,
        hardware["probe"],
        "evidence.hardware.probe",
        max_size_bytes=MAX_ROOT_JSON_BYTES,
    )
    probe = exact_object(
        read_json(probe_path, max_bytes=MAX_ROOT_JSON_BYTES),
        HARDWARE_PROBE_FIELDS,
        "evidence.hardware.probe",
    )
    require(probe["schema_version"] == 1, "hardware probe schema_version must be 1")
    require(
        probe["source_git_sha"] == source["git_sha"]
        and probe["source_tree_sha"] == source["git_tree_sha"]
        and probe["dirty_status"] == {"is_dirty": False, "status_short": []},
        "hardware probe source identity is stale",
    )
    collector = probe["collector"]
    require(isinstance(collector, dict), "hardware probe collector must be an object")
    require(
        collector
        == {
            "path": HARDWARE_PROBE_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": file_sha256(HARDWARE_PROBE_PATH),
        },
        "hardware probe collector identity is stale",
    )
    normalized = probe["normalized"]
    require(isinstance(normalized, dict), "hardware probe normalized facts must be an object")
    require(
        normalized.get("backend") == "cuda"
        and normalized.get("policy_id") == "cuda-g0-1x-rtx4090"
        and normalized.get("device_count") == 1
        and "4090" in str(normalized.get("device_name", "")),
        "hardware probe must describe exactly one RTX 4090 under the CUDA G0 policy",
    )
    fingerprint = sha256_text(probe["fingerprint"], "hardware probe fingerprint")
    require(
        fingerprint == hardware_probe.canonical_sha(normalized),
        "hardware probe fingerprint is not derived from normalized facts",
    )
    require(
        hardware["fingerprint"] == fingerprint,
        "evidence.hardware fingerprint differs from its probe",
    )

    commands = probe["commands"]
    require(isinstance(commands, list), "hardware probe commands must be a list")
    outputs: dict[str, str] = {}
    errors: dict[str, str] = {}
    seen: set[str] = set()
    for index, raw in enumerate(commands):
        label = f"evidence.hardware.probe.commands[{index}]"
        command = exact_object(raw, HARDWARE_COMMAND_FIELDS, label)
        kind = text(command["kind"], f"{label}.kind")
        require(
            kind in hardware_probe.PROBE_ARGV["cuda"] and kind not in seen,
            f"{label}.kind is invalid or duplicated",
        )
        seen.add(kind)
        require(
            command["argv"] == hardware_probe.PROBE_ARGV["cuda"][kind]
            and command["returncode"] == 0,
            f"{label} did not run the canonical CUDA probe command",
        )
        started = validate_timestamp(command["started_at"], f"{label}.started_at")
        finished = validate_timestamp(command["finished_at"], f"{label}.finished_at")
        require(
            datetime.fromisoformat(finished.replace("Z", "+00:00"))
            >= datetime.fromisoformat(started.replace("Z", "+00:00")),
            f"{label} timestamps are reversed",
        )
        number(command["duration_sec"], f"{label}.duration_sec", minimum=0.0)
        stdout_path = safe_artifact_file(
            probe_path.parent, command["stdout"], f"{label}.stdout"
        )
        stderr_path = safe_artifact_file(
            probe_path.parent, command["stderr"], f"{label}.stderr"
        )
        require(
            stdout_path.stat().st_size <= MAX_LOG_BYTES
            and stderr_path.stat().st_size <= MAX_LOG_BYTES,
            f"{label} output exceeds its byte bound",
        )
        require(
            file_sha256(stdout_path) == command["stdout_sha256"]
            and file_sha256(stderr_path) == command["stderr_sha256"],
            f"{label} output digest is stale",
        )
        outputs[kind] = stdout_path.read_text(encoding="utf-8")
        errors[kind] = stderr_path.read_text(encoding="utf-8")
    require(
        seen == set(hardware_probe.PROBE_ARGV["cuda"]),
        "hardware probe command matrix is incomplete",
    )
    try:
        recomputed = hardware_probe.normalized_from_outputs(
            "cuda",
            "cuda-g0-1x-rtx4090",
            outputs,
            errors,
        )
    except hardware_probe.ProbeError as error:
        raise DeterminismArtifactError(f"hardware probe raw output is invalid: {error}") from error
    require(
        recomputed == normalized,
        "hardware normalized facts are not derived from raw command outputs",
    )
    return fingerprint


def validate_build_command(value: Any) -> list[str]:
    require(isinstance(value, list) and value, "source.build_command must be a non-empty argv list")
    command = [text(item, f"source.build_command[{index}]") for index, item in enumerate(value)]
    require(command[:2] == ["cargo", "build"], "source.build_command must invoke cargo build")
    require("--release" in command, "source.build_command must be a release build")
    require("--features" in command, "source.build_command must declare CUDA release features")
    feature_index = command.index("--features") + 1
    require(feature_index < len(command), "source.build_command has no feature value")
    features = set(command[feature_index].split(","))
    require(
        {"cuda", "vllm-moe-marlin", "vllm-paged-attn-v2"} <= features,
        "source.build_command lacks required CUDA release features",
    )
    return command


def validate_token_shape(value: Any, phase: str, label: str) -> tuple[str, int]:
    shape = exact_object(value, TOKEN_SHAPE_FIELDS, label)
    partition = text(shape["partition"], f"{label}.partition")
    participant_count = integer(shape["participant_count"], f"{label}.participant_count", minimum=1)
    require(participant_count <= 32, f"{label}.participant_count exceeds the release bound")
    vectors: list[list[int]] = []
    for name in ("immediate_tokens", "source_start_tokens", "source_end_tokens"):
        raw = shape[name]
        require(isinstance(raw, list) and len(raw) == participant_count, f"{label}.{name} cardinality mismatch")
        vectors.append(
            [integer(item, f"{label}.{name}[{index}]") for index, item in enumerate(raw)]
        )
    immediate, starts, ends = vectors
    require(all(tokens > 0 for tokens in immediate), f"{label}.immediate_tokens must be positive")
    require(
        all(start < end and end - start == tokens for start, end, tokens in zip(starts, ends, immediate)),
        f"{label} source ranges differ from immediate token counts",
    )
    if phase == "prefill":
        require(partition in {"single_token", "multi_token", "chunk_boundary"},
                f"{label}.partition is invalid for prefill")
        if partition == "single_token":
            require(participant_count == 1 and immediate == [1] and starts == [0],
                    f"{label} single_token shape is invalid")
        elif partition == "multi_token":
            require(sum(immediate) > 1 and all(start == 0 for start in starts),
                    f"{label} multi_token shape is invalid")
        else:
            require(any(start > 0 for start in starts), f"{label} chunk_boundary lacks a non-zero source start")
    else:
        require(partition in {"c1", "multi_participant", "c32"},
                f"{label}.partition is invalid for decode")
        require(all(tokens == 1 for tokens in immediate), f"{label} decode must use one immediate token per participant")
        if partition == "c1":
            require(participant_count == 1, f"{label} c1 must contain one participant")
        elif partition == "multi_participant":
            require(1 < participant_count < 32, f"{label} multi_participant width must be 2..31")
        else:
            require(participant_count == 32, f"{label} c32 must contain 32 participants")
    return partition, participant_count


def target_key(target: dict[str, Any]) -> tuple[str, str]:
    return target["operation_id"], target["provider_id"]


def validate_target(
    value: Any,
    label: str,
    *,
    model_key: str,
    denominator: dict[str, Any],
) -> tuple[dict[str, Any], set[str], dict[str, Any]]:
    target = exact_object(value, TARGET_FIELDS, label)
    key = (
        text(target["operation_id"], f"{label}.operation_id", portable=True),
        text(target["provider_id"], f"{label}.provider_id", portable=True),
    )
    requirement = denominator["requirements"].get(key)
    require(requirement is not None, f"{label} is absent from the live coverage registry")
    for field in (
        "operation_version",
        "operation_fingerprint",
        "provider_version",
        "provider_implementation_fingerprint",
        "provider_execution_contract_fingerprint",
        "replay_equivalence",
    ):
        require(target[field] == requirement[field], f"{label}.{field} differs from live coverage")
    selection = next(
        (
            item
            for item in requirement["model_selections"]
            if item["model_key"] == model_key
        ),
        None,
    )
    require(selection is not None, f"{label} provider is not selected by model {model_key}")
    node_ids = set(validate_string_list(target["node_ids"], f"{label}.node_ids", portable=True))
    require(node_ids <= set(selection["node_ids"]), f"{label}.node_ids escape the selected provider")
    scope = denominator["scopes"].get((model_key, key[0], key[1]))
    require(scope is not None, f"{label} has no typed witness denominator")
    require(
        target["witness_plan_fingerprint"] == scope["witness_plan_fingerprint"],
        f"{label}.witness_plan_fingerprint is stale",
    )
    return target, node_ids, scope


def witness_topology(witness: dict[str, Any]) -> tuple[Any, ...]:
    return (
        witness["kind"],
        witness["semantic_id"],
        witness["node_id"],
        witness["resource_id"],
        witness["access"],
        witness["participant_index"],
        witness["logical_offset_bytes"],
        witness["length_bytes"],
        witness["element_type"],
    )


def witness_static_topology(witness: dict[str, Any]) -> tuple[Any, ...]:
    return (
        witness["kind"],
        witness["semantic_id"],
        witness["node_id"],
        witness["resource_id"],
        witness["access"],
        witness["participant_index"],
        witness["element_type"],
    )


def validate_runtime_witness_range(
    witness: dict[str, Any],
    expected: dict[str, Any],
    token_shape: dict[str, Any],
    label: str,
) -> None:
    location = expected["location"]
    extent = location["extent"]
    participant = witness["participant_index"]
    offset = witness["logical_offset_bytes"]
    length = witness["length_bytes"]
    kind = extent["kind"]
    if kind == "fixed":
        require(
            offset == location["logical_offset_bytes"]
            and length == location["declared_length_bytes"],
            f"{label} fixed witness range differs from the immutable plan",
        )
        return
    if kind == "immediate_token_span":
        bytes_per_token = extent["bytes_per_token"]
        expected_length = bytes_per_token * token_shape["immediate_tokens"][participant]
        packed_start = sum(token_shape["immediate_tokens"][:participant])
        allowed_offsets = {
            bytes_per_token * packed_start,
            bytes_per_token * token_shape["source_start_tokens"][participant],
        }
        require(
            offset in allowed_offsets and length == expected_length,
            f"{label} immediate-token witness range differs from prepared work",
        )
        return
    bytes_per_token = extent["bytes_per_token"]
    minimum_length = bytes_per_token * token_shape["source_end_tokens"][participant]
    require(
        offset == 0
        and minimum_length <= length <= extent["maximum_storage_bytes"],
        f"{label} active state prefix differs from prepared work or immutable capacity",
    )


def validate_execution(
    value: Any,
    label: str,
    *,
    target_nodes: set[str],
    participant_count: int,
    expected_witnesses: list[dict[str, Any]],
    token_shape: dict[str, Any],
) -> tuple[dict[str, Any], dict[tuple[Any, ...], str]]:
    execution = exact_object(value, EXECUTION_FIELDS, label)
    text(execution["execution_id"], f"{label}.execution_id", portable=True)
    mode = text(execution["mode"], f"{label}.mode")
    require(mode in {"eager", "replay"}, f"{label}.mode is invalid")
    sha256_text(execution["restore_sha256"], f"{label}.restore_sha256")
    submission_fingerprint = sha256_text(
        execution["submission_fingerprint"], f"{label}.submission_fingerprint"
    )
    sha256_text(execution["receipt_fingerprint"], f"{label}.receipt_fingerprint")

    attribution = exact_object(execution["attribution"], ATTRIBUTION_FIELDS, f"{label}.attribution")
    sha256_text(
        attribution["batch_identity_fingerprint"],
        f"{label}.attribution.batch_identity_fingerprint",
    )
    require(
        attribution["submission_fingerprint"] == submission_fingerprint,
        f"{label}.attribution submission fingerprint differs from execution",
    )
    reusable_fingerprint = attribution["reusable_executable_fingerprint"]
    if mode == "replay":
        sha256_text(reusable_fingerprint, f"{label}.attribution.reusable_executable_fingerprint")
    else:
        require(reusable_fingerprint is None, f"{label} eager execution has reusable attribution")

    commands_raw = attribution["commands"]
    require(isinstance(commands_raw, list) and commands_raw, f"{label}.attribution.commands must be non-empty")
    compute_nodes: set[str] = set()
    for index, raw in enumerate(commands_raw):
        command_label = f"{label}.attribution.commands[{index}]"
        command = exact_object(raw, COMMAND_FIELDS, command_label)
        require(command["command_index"] == index, f"{command_label}.command_index is not contiguous")
        node_id = command["node_id"]
        if node_id is not None:
            text(node_id, f"{command_label}.node_id", portable=True)
        phase = text(command["command_phase"], f"{command_label}.command_phase")
        require(
            phase in {"initialization", "dynamic_binding", "compute", "result_binding"},
            f"{command_label}.command_phase is invalid",
        )
        text(command["native_op_id"], f"{command_label}.native_op_id", portable=True)
        execution_path = text(command["execution_path"], f"{command_label}.execution_path")
        require(execution_path in {"eager", "replayed"}, f"{command_label}.execution_path is invalid")
        text(command["batching_form"], f"{command_label}.batching_form")
        command_participants = integer(
            command["participant_count"], f"{command_label}.participant_count"
        )
        integer(command["token_count"], f"{command_label}.token_count")
        compute_dispatch_count = integer(
            command["compute_dispatch_count"], f"{command_label}.compute_dispatch_count"
        )
        transfer_count = integer(
            command["transfer_command_count"], f"{command_label}.transfer_command_count"
        )
        require(
            compute_dispatch_count > 0 or transfer_count > 0,
            f"{command_label} contains no native work",
        )
        graph_nodes = command["reusable_graph_node_count"]
        if graph_nodes is not None:
            integer(graph_nodes, f"{command_label}.reusable_graph_node_count", minimum=1)
        if phase == "compute" and node_id in target_nodes:
            require(
                command_participants == participant_count,
                f"{command_label} participant count differs from the case",
            )
            if mode == "eager":
                require(
                    execution_path == "eager" and graph_nodes is None,
                    f"{command_label} eager proof did not use the eager path",
                )
            else:
                require(
                    execution_path == "replayed" and graph_nodes is not None,
                    f"{command_label} replay proof used eager fallback or lacks graph attribution",
                )
            compute_nodes.add(node_id)
    require(compute_nodes == target_nodes, f"{label} attribution does not cover every target node")

    witnesses_raw = execution["witnesses"]
    require(
        isinstance(witnesses_raw, list)
        and 0 < len(witnesses_raw) <= MAX_WITNESSES_PER_EXECUTION,
        f"{label}.witnesses cardinality is invalid",
    )
    witnesses: dict[tuple[Any, ...], str] = {}
    static_witnesses: dict[tuple[Any, ...], dict[str, Any]] = {}
    witnessed_nodes: set[str] = set()
    output_count = 0
    for index, raw in enumerate(witnesses_raw):
        witness_label = f"{label}.witnesses[{index}]"
        witness = exact_object(raw, WITNESS_FIELDS, witness_label)
        kind = text(witness["kind"], f"{witness_label}.kind")
        require(kind in {"declared_output", "state_effect"}, f"{witness_label}.kind is invalid")
        text(witness["semantic_id"], f"{witness_label}.semantic_id", portable=True)
        node_id = text(witness["node_id"], f"{witness_label}.node_id", portable=True)
        require(node_id in target_nodes, f"{witness_label}.node_id is not a target node")
        text(witness["resource_id"], f"{witness_label}.resource_id", portable=True)
        access = text(witness["access"], f"{witness_label}.access")
        if kind == "declared_output":
            require(access == "write", f"{witness_label} declared output must be write")
            output_count += 1
        else:
            require(access in {"write", "read_write"}, f"{witness_label} state effect has invalid access")
        participant_index = integer(
            witness["participant_index"], f"{witness_label}.participant_index"
        )
        require(participant_index < participant_count, f"{witness_label}.participant_index is out of range")
        integer(witness["logical_offset_bytes"], f"{witness_label}.logical_offset_bytes")
        integer(witness["length_bytes"], f"{witness_label}.length_bytes", minimum=1)
        text(witness["element_type"], f"{witness_label}.element_type", portable=True)
        digest = sha256_text(witness["raw_sha256"], f"{witness_label}.raw_sha256")
        topology = witness_topology(witness)
        require(topology not in witnesses, f"{witness_label} duplicates another witness")
        witnesses[topology] = digest
        static_topology = witness_static_topology(witness)
        require(
            static_topology not in static_witnesses,
            f"{witness_label} duplicates a semantic participant witness",
        )
        static_witnesses[static_topology] = witness
        witnessed_nodes.add(node_id)
    require(output_count > 0, f"{label} contains no declared node output witness")
    require(witnessed_nodes == target_nodes, f"{label} does not witness every target node")
    require(
        list(witnesses) == sorted(witnesses),
        f"{label}.witnesses must use canonical semantic/range order",
    )
    expected_by_static: dict[tuple[Any, ...], dict[str, Any]] = {}
    for expected in expected_witnesses:
        if expected["node_id"] not in target_nodes:
            continue
        for participant_index in range(participant_count):
            key = (
                expected["kind"],
                expected["semantic_id"],
                expected["node_id"],
                expected["resource_id"],
                expected["access"],
                participant_index,
                expected["element_type"],
            )
            require(key not in expected_by_static, f"{label} typed witness denominator duplicates {key}")
            expected_by_static[key] = expected
    require(
        set(static_witnesses) == set(expected_by_static),
        f"{label} runtime witnesses differ from the typed witness denominator",
    )
    for key, witness in static_witnesses.items():
        validate_runtime_witness_range(
            witness,
            expected_by_static[key],
            token_shape,
            f"{label}.witness[{key}]",
        )
    return execution, witnesses


def validate_case(
    case: dict[str, Any],
    label: str,
    *,
    root_identity: dict[str, Any],
    denominator: dict[str, Any],
) -> dict[str, Any]:
    case = exact_object(case, CASE_FIELDS, label)
    require(case["schema_version"] == 1, f"{label}.schema_version must be 1")
    case_id = text(case["case_id"], f"{label}.case_id", portable=True)
    require(
        case["denominator_fingerprint"] == root_identity["denominator_fingerprint"],
        f"{label}.denominator_fingerprint is stale",
    )
    require(case["binary_sha256"] == root_identity["binary_sha256"],
            f"{label}.binary_sha256 is stale")
    require(
        case["device_runtime_implementation_fingerprint"]
        == denominator["runtime_fingerprint"],
        f"{label}.device runtime fingerprint is stale",
    )
    require(case["device_fingerprint"] == root_identity["device_fingerprint"],
            f"{label}.device_fingerprint is stale")
    model_key = text(case["model_key"], f"{label}.model_key", portable=True)
    require(
        model_key in denominator["models"],
        f"{label}.model_key is absent from denominator",
    )
    model = denominator["models"][model_key]
    require(
        case["resolved_plan_fingerprint"] == model["resolved_plan_fingerprint"]
        and case["plan_hash"] == model["plan_hash"],
        f"{label} differs from the resolved model plan",
    )
    phase = text(case["phase"], f"{label}.phase")
    require(phase in {"prefill", "decode"}, f"{label}.phase is invalid")
    partition, participant_count = validate_token_shape(
        case["token_shape"], phase, f"{label}.token_shape"
    )
    text(case["dtype"], f"{label}.dtype", portable=True)
    text(case["quantization"], f"{label}.quantization", portable=True)

    initialization = exact_object(
        case["initialization"], INITIALIZATION_FIELDS, f"{label}.initialization"
    )
    for field in ("input_sha256", "rng_sha256", "initial_state_sha256"):
        sha256_text(initialization[field], f"{label}.initialization.{field}")
    initial_state_kind = text(
        initialization["initial_state_kind"], f"{label}.initialization.initial_state_kind"
    )
    require(
        initial_state_kind in {"none", "zero", "nonzero"},
        f"{label}.initialization.initial_state_kind is invalid",
    )
    poison = text(initialization["workspace_poison"], f"{label}.initialization.workspace_poison")
    require(poison in {"00", "a5"}, f"{label}.initialization.workspace_poison is invalid")

    targets_raw = case["coverage_targets"]
    require(isinstance(targets_raw, list) and targets_raw, f"{label}.coverage_targets must be non-empty")
    expected_witnesses: list[dict[str, Any]] = []
    target_nodes: set[str] = set()
    target_node_coverage: dict[tuple[str, str], set[str]] = {}
    replay_equivalence: str | None = None
    previous_target_key: tuple[str, str] | None = None
    for index, raw in enumerate(targets_raw):
        target, nodes, scope = validate_target(
            raw,
            f"{label}.coverage_targets[{index}]",
            model_key=model_key,
            denominator=denominator,
        )
        key = target_key(target)
        require(previous_target_key is None or previous_target_key < key,
                f"{label}.coverage_targets are not canonical")
        previous_target_key = key
        require(not (target_nodes & nodes), f"{label}.coverage_targets overlap nodes")
        target_nodes.update(nodes)
        target_node_coverage[key] = nodes
        if replay_equivalence is None:
            replay_equivalence = target["replay_equivalence"]
        require(
            replay_equivalence == target["replay_equivalence"],
            f"{label} mixes replay-eligible and eager-only providers",
        )
        expected_witnesses.extend(scope["expected_runtime_witnesses"])

    executions_raw = case["executions"]
    require(
        isinstance(executions_raw, list)
        and 0 < len(executions_raw) <= MAX_EXECUTIONS_PER_CASE,
        f"{label}.executions cardinality is invalid",
    )
    executions: dict[str, dict[str, Any]] = {}
    witness_maps: dict[str, dict[tuple[Any, ...], str]] = {}
    canonical_topology: set[tuple[Any, ...]] | None = None
    replay_fingerprint: str | None = None
    restore_fingerprint: str | None = None
    for index, raw in enumerate(executions_raw):
        execution, witnesses = validate_execution(
            raw,
            f"{label}.executions[{index}]",
            target_nodes=target_nodes,
            participant_count=participant_count,
            expected_witnesses=expected_witnesses,
            token_shape=case["token_shape"],
        )
        execution_id = execution["execution_id"]
        require(execution_id not in executions, f"{label} duplicates execution {execution_id}")
        topology = set(witnesses)
        if canonical_topology is None:
            canonical_topology = topology
        require(topology == canonical_topology, f"{label} execution witness topology drifted")
        if execution["mode"] == "replay":
            current = execution["attribution"]["reusable_executable_fingerprint"]
            if replay_fingerprint is None:
                replay_fingerprint = current
            require(current == replay_fingerprint, f"{label} replay executable fingerprint drifted")
        current_restore = execution["restore_sha256"]
        if restore_fingerprint is None:
            restore_fingerprint = current_restore
        require(
            current_restore == restore_fingerprint,
            f"{label} executions do not restore the same input/RNG/initial-state bytes",
        )
        executions[execution_id] = execution
        witness_maps[execution_id] = witnesses
    require(
        list(executions) == sorted(executions),
        f"{label}.executions must use canonical execution-id order",
    )

    comparisons_raw = case["comparisons"]
    require(isinstance(comparisons_raw, list), f"{label}.comparisons must be a list")
    expected_kinds = (
        {"eager_eager"}
        if replay_equivalence == "ineligible"
        else {"eager_eager", "replay_replay", "eager_replay"}
    )
    comparisons_by_kind: dict[str, list[dict[str, Any]]] = {}
    used_executions: set[str] = set()
    pair_keys: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(comparisons_raw):
        comparison_label = f"{label}.comparisons[{index}]"
        comparison = exact_object(raw, COMPARISON_FIELDS, comparison_label)
        kind = text(comparison["kind"], f"{comparison_label}.kind")
        require(kind in expected_kinds, f"{comparison_label}.kind is not required by coverage")
        ordinal = integer(comparison["ordinal"], f"{comparison_label}.ordinal")
        left_id = text(comparison["left_execution_id"], f"{comparison_label}.left_execution_id", portable=True)
        right_id = text(comparison["right_execution_id"], f"{comparison_label}.right_execution_id", portable=True)
        require(left_id != right_id, f"{comparison_label} compares one execution to itself")
        require(left_id in executions and right_id in executions, f"{comparison_label} references an unknown execution")
        expected_modes = {
            "eager_eager": ("eager", "eager"),
            "replay_replay": ("replay", "replay"),
            "eager_replay": ("eager", "replay"),
        }[kind]
        require(
            (executions[left_id]["mode"], executions[right_id]["mode"]) == expected_modes,
            f"{comparison_label} execution modes do not match {kind}",
        )
        require(comparison["relation"] == "bitwise_equal", f"{comparison_label} uses a non-exact relation")
        require(comparison["first_mismatch"] is None, f"{comparison_label} records a mismatch")
        pair_key = (kind, left_id, right_id)
        require(pair_key not in pair_keys, f"{comparison_label} duplicates a comparison pair")
        pair_keys.add(pair_key)
        for topology, left_digest in witness_maps[left_id].items():
            require(
                left_digest == witness_maps[right_id][topology],
                f"{comparison_label} raw witness mismatch at {topology}",
            )
        comparisons_by_kind.setdefault(kind, []).append(comparison)
        used_executions.update((left_id, right_id))
    require(set(comparisons_by_kind) == expected_kinds, f"{label} is missing a required comparison kind")
    for kind, comparisons in comparisons_by_kind.items():
        require(
            5 <= len(comparisons) <= MAX_COMPARISONS_PER_KIND,
            f"{label} {kind} comparison count is outside 5..{MAX_COMPARISONS_PER_KIND}",
        )
        require(
            [item["ordinal"] for item in comparisons] == list(range(len(comparisons))),
            f"{label} {kind} comparison ordinals are not contiguous",
        )
    require(used_executions == set(executions), f"{label} contains unused executions")
    require(case["first_mismatch"] is None, f"{label}.first_mismatch must be null for PASS evidence")

    state_witness = any(
        topology[0] == "state_effect"
        for topology in (canonical_topology or set())
    )
    if initial_state_kind in {"zero", "nonzero"}:
        require(state_witness, f"{label} state fixture has no state-effect witness")
    return {
        "case_id": case_id,
        "model_key": model_key,
        "phase": phase,
        "partition": partition,
        "initial_state_kind": initial_state_kind,
        "workspace_poison": poison,
        "target_node_coverage": target_node_coverage,
        "target_keys": set(target_node_coverage),
        "target_nodes": target_nodes,
        "state_witness": state_witness,
        "execution_count": len(executions),
        "comparison_count": len(comparisons_raw),
        "witness_count": len(canonical_topology or set()) * len(executions),
    }


def validate_artifact(root: Path, expected_source: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    require(root.is_dir() and not root.is_symlink(), "artifact root must be a real directory")
    manifest_path = root / "evidence.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "missing evidence.json")
    manifest = exact_object(
        read_json(manifest_path, max_bytes=MAX_ROOT_JSON_BYTES),
        ROOT_FIELDS,
        "evidence",
    )
    require(manifest["schema_version"] == 1, "evidence.schema_version must be 1")
    require(manifest["artifact_type"] == ARTIFACT_TYPE, "evidence.artifact_type is invalid")
    require(manifest["backend"] == "cuda", "evidence.backend must be cuda")

    source = exact_object(manifest["source"], SOURCE_FIELDS, "evidence.source")
    git_sha = text(source["git_sha"], "evidence.source.git_sha")
    require(GIT_SHA_RE.fullmatch(git_sha) is not None, "evidence.source.git_sha is invalid")
    git_tree_sha = text(source["git_tree_sha"], "evidence.source.git_tree_sha")
    require(
        GIT_SHA_RE.fullmatch(git_tree_sha) is not None,
        "evidence.source.git_tree_sha is invalid",
    )
    require(
        git_sha == expected_source["git_sha"]
        and git_tree_sha == expected_source["git_tree_sha"],
        "evidence.source commit or tree identity is stale",
    )
    require(source["dirty_status"] == [], "evidence.source.dirty_status must be empty")
    validate_build_command(source["build_command"])
    text(source["binary_path"], "evidence.source.binary_path")
    binary_ref, binary_path = validate_file_ref(
        root,
        source["binary"],
        "evidence.source.binary",
        max_size_bytes=MAX_BINARY_BYTES,
    )
    require(
        binary_path.stat().st_mode & 0o111 != 0,
        "evidence.source.binary must preserve an executable mode",
    )
    binary_sha256 = binary_ref["sha256"]

    device_fingerprint = validate_hardware(root, manifest["hardware"], source=source)
    locked_models = validate_models_lock(root, manifest["models_lock"])

    denominator_ref = exact_object(
        manifest["denominator"],
        DENOMINATOR_REF_FIELDS,
        "evidence.denominator",
    )
    denominator_file_ref = {key: denominator_ref[key] for key in FILE_REF_FIELDS}
    _, denominator_path = validate_file_ref(
        root,
        denominator_file_ref,
        "evidence.denominator",
        max_size_bytes=MAX_DENOMINATOR_JSON_BYTES,
    )
    denominator_fingerprint = sha256_text(
        denominator_ref["fingerprint"], "evidence.denominator.fingerprint"
    )
    require(
        denominator_fingerprint == denominator_ref["sha256"],
        "evidence.denominator fingerprint differs from the exact Rust bytes",
    )
    denominator = validate_denominator(
        read_json(denominator_path, max_bytes=MAX_DENOMINATOR_JSON_BYTES)
    )

    models_raw = manifest["models"]
    require(isinstance(models_raw, list), "evidence.models must be a list")
    model_keys: list[str] = []
    for index, raw in enumerate(models_raw):
        label = f"evidence.models[{index}]"
        model = exact_object(raw, MODEL_FIELDS, label)
        model_key = text(model["model_key"], f"{label}.model_key", portable=True)
        require(
            model_key in denominator["models"],
            f"{label}.model_key is absent from denominator",
        )
        denominator_model = denominator["models"][model_key]
        locked_model = locked_models[model_key]
        text(model["source_model_id"], f"{label}.source_model_id", portable=True)
        revision = text(model["revision"], f"{label}.revision", portable=True)
        require(
            GIT_SHA_RE.fullmatch(revision) is not None,
            f"{label}.revision must be an immutable commit",
        )
        files = model["files"]
        require(isinstance(files, list) and files, f"{label}.files must be non-empty")
        file_paths: list[str] = []
        for file_index, file_raw in enumerate(files):
            file_label = f"{label}.files[{file_index}]"
            model_file = exact_object(file_raw, MODEL_FILE_FIELDS, file_label)
            file_paths.append(text(model_file["path"], f"{file_label}.path"))
            sha256_text(model_file["sha256"], f"{file_label}.sha256")
            integer(model_file["size_bytes"], f"{file_label}.size_bytes", minimum=1)
        require(file_paths == sorted(set(file_paths)), f"{label}.files must be sorted and unique")
        sha256_text(model["config_sha256"], f"{label}.config_sha256")
        require(
            {
                "model_key": model_key,
                "source_model_id": model["source_model_id"],
                "revision": model["revision"],
                "files": model["files"],
                "config_sha256": model["config_sha256"],
            }
            == locked_model,
            f"{label} differs from models.lock",
        )
        require(
            model["external_metadata_id"] == denominator_model["external_metadata_id"]
            and model["resolved_plan_fingerprint"]
            == denominator_model["resolved_plan_fingerprint"]
            and model["plan_hash"] == denominator_model["plan_hash"],
            f"{label} differs from the live denominator model identity",
        )
        model_keys.append(model_key)
    require(model_keys == sorted(PRIMARY_MODELS), "evidence.models must contain the three primary models")

    runner = exact_object(manifest["runner"], RUNNER_FIELDS, "evidence.runner")
    require(isinstance(runner["command"], list) and runner["command"], "evidence.runner.command must be non-empty")
    runner_command = [
        text(item, f"evidence.runner.command[{index}]")
        for index, item in enumerate(runner["command"])
    ]
    require(
        runner_command[0] == source["binary_path"],
        "evidence.runner.command must execute the recorded release binary",
    )
    require(
        len(runner_command) == 12
        and runner_command[1] == "vnext-determinism"
        and runner_command[2] == "--models-lock"
        and runner_command[4] == "--artifact-root"
        and runner_command[6::2] == ["--model", "--model", "--model"],
        "evidence.runner.command is not the canonical vnext-determinism collector command",
    )
    model_arguments = runner_command[7::2]
    require(
        sorted(argument.split("=", 1)[0] for argument in model_arguments)
        == sorted(PRIMARY_MODELS)
        and all("=" in argument and argument.split("=", 1)[1] for argument in model_arguments),
        "evidence.runner.command must bind exactly the three primary model directories",
    )
    started = validate_timestamp(runner["started_at"], "evidence.runner.started_at")
    finished = validate_timestamp(runner["finished_at"], "evidence.runner.finished_at")
    require(
        datetime.fromisoformat(finished.replace("Z", "+00:00"))
        > datetime.fromisoformat(started.replace("Z", "+00:00")),
        "evidence.runner timestamps are not increasing",
    )
    require(runner["exit_code"] == 0, "evidence.runner.exit_code must be 0")
    _, receipt_path = validate_file_ref(
        root,
        runner["receipt"],
        "evidence.runner.receipt",
        max_size_bytes=MAX_RECEIPT_JSON_BYTES,
    )
    stdout_ref, _ = validate_file_ref(
        root,
        runner["stdout"],
        "evidence.runner.stdout",
        max_size_bytes=MAX_LOG_BYTES,
    )
    stderr_ref, _ = validate_file_ref(
        root,
        runner["stderr"],
        "evidence.runner.stderr",
        max_size_bytes=MAX_LOG_BYTES,
    )
    validate_bounded_receipt(
        read_json(receipt_path, max_bytes=MAX_RECEIPT_JSON_BYTES),
        command=runner_command,
        started_at=started,
        finished_at=finished,
        exit_code=runner["exit_code"],
        stdout_ref=stdout_ref,
        stderr_ref=stderr_ref,
    )

    case_refs = manifest["cases"]
    require(
        isinstance(case_refs, list) and 0 < len(case_refs) <= MAX_CASES,
        "evidence.cases cardinality is invalid",
    )
    case_paths: list[str] = []
    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    root_identity = {
        "denominator_fingerprint": denominator_fingerprint,
        "binary_sha256": binary_sha256,
        "device_fingerprint": device_fingerprint,
    }
    for index, raw_ref in enumerate(case_refs):
        ref, case_path = validate_file_ref(
            root,
            raw_ref,
            f"evidence.cases[{index}]",
            max_size_bytes=MAX_CASE_JSON_BYTES,
        )
        case_paths.append(ref["path"])
        case = validate_case(
            read_json(case_path, max_bytes=MAX_CASE_JSON_BYTES),
            f"case[{index}]",
            root_identity=root_identity,
            denominator=denominator,
        )
        require(case["case_id"] not in case_ids, "evidence contains duplicate case ids")
        case_ids.add(case["case_id"])
        cases.append(case)
    require(case_paths == sorted(set(case_paths)), "evidence.cases must be sorted and unique")
    actual_case_paths = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "cases").glob("*.json")
        if path.is_file() and not path.is_symlink()
    )
    require(actual_case_paths == case_paths, "case file set differs from evidence references")

    partitions_by_model: dict[str, set[tuple[str, str]]] = {
        key: set() for key in PRIMARY_MODELS
    }
    state_by_model: dict[str, set[str]] = {key: set() for key in PRIMARY_MODELS}
    coverage_nodes: dict[tuple[str, tuple[str, str]], set[str]] = {}
    poisons: dict[tuple[str, tuple[str, str]], set[str]] = {}
    phases: dict[tuple[str, tuple[str, str]], set[str]] = {}
    for case in cases:
        model_key = case["model_key"]
        partitions_by_model[model_key].add((case["phase"], case["partition"]))
        if case["state_witness"] and case["initial_state_kind"] in {"zero", "nonzero"}:
            state_by_model[model_key].add(case["initial_state_kind"])
        for requirement_key, nodes in case["target_node_coverage"].items():
            selection_key = (model_key, requirement_key)
            coverage_nodes.setdefault(selection_key, set()).update(nodes)
            poisons.setdefault(selection_key, set()).add(case["workspace_poison"])
            phases.setdefault(selection_key, set()).add(case["phase"])

    for model_key in PRIMARY_MODELS:
        require(
            partitions_by_model[model_key] == REQUIRED_PARTITIONS,
            f"model {model_key} shape partition coverage is incomplete",
        )
        require(
            state_by_model[model_key] == {"zero", "nonzero"},
            f"model {model_key} lacks zero/nonzero state-effect evidence",
        )
    fixture_matrix: dict[
        tuple[str, tuple[str, str], str, str],
        set[tuple[str, str]],
    ] = {}
    for case in cases:
        for requirement_key in case["target_keys"]:
            fixture_key = (
                case["model_key"],
                requirement_key,
                case["phase"],
                case["partition"],
            )
            fixture_matrix.setdefault(fixture_key, set()).add(
                (case["initial_state_kind"], case["workspace_poison"])
            )
    for requirement_key, requirement in denominator["requirements"].items():
        for selection in requirement["model_selections"]:
            selection_key = (selection["model_key"], requirement_key)
            require(
                coverage_nodes.get(selection_key) == set(selection["node_ids"]),
                f"provider selection {selection_key} lacks exact node proof coverage",
            )
            require(
                poisons.get(selection_key) == {"00", "a5"},
                f"provider selection {selection_key} lacks both workspace poison patterns",
            )
            require(
                phases.get(selection_key) == {"prefill", "decode"},
                f"provider selection {selection_key} lacks prefill/decode proof",
            )
            for phase, partition in REQUIRED_PARTITIONS:
                require(
                    fixture_matrix.get(
                        (selection["model_key"], requirement_key, phase, partition)
                    )
                    == {
                        ("zero", "00"),
                        ("zero", "a5"),
                        ("nonzero", "00"),
                        ("nonzero", "a5"),
                    },
                    f"provider selection {selection_key} lacks the exact state/poison cross-product for {phase}/{partition}",
                )

    return {
        "source_git_sha": git_sha,
        "source_tree_sha": git_tree_sha,
        "binary_sha256": binary_sha256,
        "denominator_fingerprint": denominator_fingerprint,
        "device_fingerprint": device_fingerprint,
        "model_keys": sorted(PRIMARY_MODELS),
        "provider_requirement_count": len(denominator["requirements"]),
        "case_count": len(cases),
        "execution_count": sum(case["execution_count"] for case in cases),
        "comparison_count": sum(case["comparison_count"] for case in cases),
        "witness_count": sum(case["witness_count"] for case in cases),
        "input_manifest_sha256": file_sha256(manifest_path),
    }


def git_stdout(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        result.returncode == 0,
        f"cannot resolve current git {' '.join(args)}: {result.stderr.strip()}",
    )
    return result.stdout.strip()


def current_git_state() -> dict[str, str]:
    git_sha = git_stdout("rev-parse", "HEAD")
    git_tree_sha = git_stdout("rev-parse", "HEAD^{tree}")
    status = git_stdout("status", "--short", "--untracked-files=no")
    require(GIT_SHA_RE.fullmatch(git_sha) is not None, "current git SHA is invalid")
    require(GIT_SHA_RE.fullmatch(git_tree_sha) is not None, "current git tree SHA is invalid")
    require(
        not status,
        "current tracked worktree must be clean before validating CUDA evidence",
    )
    return {"git_sha": git_sha, "git_tree_sha": git_tree_sha}


def validation_manifest(
    artifact_root: Path,
    out_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    pass_line = f"{PASS_PREFIX}: {out_dir}"
    return {
        "schema_version": 1,
        "artifact_type": VALIDATOR_ARTIFACT_TYPE,
        "lane": "runtime-vnext-cuda-determinism",
        "status": "pass",
        "pass_line": pass_line,
        "artifact_dir": str(out_dir),
        "input_artifact_root": str(artifact_root.resolve()),
        **summary,
    }


def rejection_failure_class(error: Exception) -> str:
    message = str(error).lower()
    if "worktree" in message or "source" in message or "git" in message:
        return "source_identity"
    if "hardware" in message or "rtx 4090" in message:
        return "hardware_identity"
    if "models.lock" in message or "model catalog" in message:
        return "model_identity"
    if "denominator" in message or "coverage" in message or "witness plan" in message:
        return "determinism_denominator"
    if "runner" in message or "receipt" in message:
        return "runner_containment"
    if "mismatch" in message or "bitwise" in message:
        return "correctness_mismatch"
    return "artifact_contract"


def write_rejection_manifest(
    artifact_root: Path,
    out_dir: Path,
    error: Exception,
) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.json"
        if manifest_path.exists() or any(out_dir.iterdir()):
            return
        write_json(
            manifest_path,
            {
                "schema_version": 1,
                "artifact_type": VALIDATOR_ARTIFACT_TYPE,
                "lane": "runtime-vnext-cuda-determinism",
                "status": "reject",
                "failure_class": rejection_failure_class(error),
                "message": str(error),
                "artifact_dir": str(out_dir),
                "input_artifact_root": str(artifact_root.resolve()),
                "recorded_at": datetime.now().astimezone().isoformat(),
            },
            exclusive=True,
        )
    except OSError:
        return


def selftest_location(
    model_key: str,
    suffix: str,
    *,
    kind: str,
    index: int,
) -> dict[str, Any]:
    node_id = f"node.{model_key}.{suffix}"
    if kind == "output":
        return {
            "node_id": node_id,
            "value_id": f"value.output.{index}",
            "role": "output",
            "ordinal": 0,
            "usage": "result",
            "storage_component_ordinal": 0,
            "storage_component_id": None,
            "resource_id": f"resource.output.{index}",
            "logical_offset_bytes": 0,
            "declared_length_bytes": 4,
            "element_type": "f32",
            "extent": {"kind": "fixed"},
        }
    return {
        "node_id": node_id,
        "value_id": f"value.state.{index}",
        "role": "state",
        "ordinal": 0,
        "usage": "state",
        "storage_component_ordinal": 0,
        "storage_component_id": None,
        "resource_id": f"resource.state.{index}",
        "logical_offset_bytes": 0,
        "declared_length_bytes": 4,
        "element_type": "f32",
        "extent": {"kind": "fixed"},
    }


def selftest_witness_plan(
    model: dict[str, Any],
    requirement: dict[str, Any],
    selection: dict[str, Any],
    *,
    suffix: str,
    index: int,
) -> dict[str, Any]:
    node_id = selection["node_ids"][0]
    output_location = selftest_location(
        model["model_key"], suffix, kind="output", index=index
    )
    state_location = selftest_location(
        model["model_key"], suffix, kind="state", index=index
    )
    return {
        "schema_version": {"major": 4, "minor": 0},
        "plan_hash": model["plan_hash"],
        "node_ids": [node_id],
        "replay_provider_requirements": [
            {
                "provider_id": requirement["provider_id"],
                "provider_implementation_fingerprint": requirement[
                    "provider_implementation_fingerprint"
                ],
                "provider_execution_contract_fingerprint": requirement[
                    "provider_execution_contract_fingerprint"
                ],
                "node_ids": [node_id],
            }
        ],
        "initializations": [
            {
                "kind": {
                    "kind": "state",
                    "state_id": f"state.cache.{index}",
                    "state_value_id": f"value.state.{index}",
                    "lifetime": "request",
                    "access": "read_write",
                },
                "location": state_location,
                "consumer_node_ids": [node_id],
            }
        ],
        "witnesses": [
            {
                "provider_id": requirement["provider_id"],
                "provider_implementation_fingerprint": requirement[
                    "provider_implementation_fingerprint"
                ],
                "provider_execution_contract_fingerprint": requirement[
                    "provider_execution_contract_fingerprint"
                ],
                "kind": {
                    "kind": "output",
                    "value_id": f"value.output.{index}",
                    "output_ordinal": 0,
                },
                "location": output_location,
            },
            {
                "provider_id": requirement["provider_id"],
                "provider_implementation_fingerprint": requirement[
                    "provider_implementation_fingerprint"
                ],
                "provider_execution_contract_fingerprint": requirement[
                    "provider_execution_contract_fingerprint"
                ],
                "kind": {
                    "kind": "state_effect",
                    "state_id": f"state.cache.{index}",
                    "state_value_id": f"value.state.{index}",
                    "lifetime": "request",
                    "access": "read_write",
                },
                "location": state_location,
            },
        ],
    }


def make_selftest_denominator(
    root: Path,
) -> tuple[dict[str, Any], str, dict[tuple[str, str], str]]:
    models = []
    primary_selections = []
    secondary_selections = []
    for index, model_key in enumerate(sorted(PRIMARY_MODELS), 1):
        node_ids = [
            f"node.{model_key}.primary",
            f"node.{model_key}.secondary",
        ]
        resolved = f"{index}" * 64
        plan_hash = f"{index + 3}" * 64
        models.append(
            {
                "model_key": model_key,
                "external_metadata_id": f"metadata.{model_key}",
                "resolved_plan_fingerprint": resolved,
                "plan_hash": plan_hash,
                "node_ids": node_ids,
            }
        )
        primary_selections.append(
            {
                "model_key": model_key,
                "resolved_plan_fingerprint": resolved,
                "plan_hash": plan_hash,
                "node_ids": [node_ids[0]],
            }
        )
        secondary_selections.append(
            {
                "model_key": model_key,
                "resolved_plan_fingerprint": resolved,
                "plan_hash": plan_hash,
                "node_ids": [node_ids[1]],
            }
        )
    requirements = [
        {
            "operation_id": "operation.test",
            "operation_version": {"major": 1, "minor": 0},
            "operation_fingerprint": "c" * 64,
            "provider_id": "provider.cuda.test",
            "provider_version": {"major": 1, "minor": 0},
            "provider_implementation_fingerprint": "d" * 64,
            "provider_execution_contract_fingerprint": "e" * 64,
            "replay_equivalence": "bitwise_eager_equivalent",
            "required_comparisons": [
                "eager_eager",
                "replay_replay",
                "eager_replay",
            ],
            "model_selections": primary_selections,
        },
        {
            "operation_id": "operation.test.secondary",
            "operation_version": {"major": 1, "minor": 0},
            "operation_fingerprint": "1" * 64,
            "provider_id": "provider.cuda.test.secondary",
            "provider_version": {"major": 1, "minor": 0},
            "provider_implementation_fingerprint": "2" * 64,
            "provider_execution_contract_fingerprint": "3" * 64,
            "replay_equivalence": "bitwise_eager_equivalent",
            "required_comparisons": [
                "eager_eager",
                "replay_replay",
                "eager_replay",
            ],
            "model_selections": secondary_selections,
        },
    ]
    coverage = {
        "schema_version": {"major": 1, "minor": 0},
        "device_id": "device.cuda.0",
        "device_runtime_implementation_fingerprint": "a" * 64,
        "capability_catalog_fingerprint": "b" * 64,
        "models": models,
        "provider_requirements": requirements,
    }
    provider_evidence = []
    witness_fingerprints: dict[tuple[str, str], str] = {}
    for model in models:
        for index, (requirement, suffix) in enumerate(
            zip(requirements, ("primary", "secondary"))
        ):
            selection = next(
                item
                for item in requirement["model_selections"]
                if item["model_key"] == model["model_key"]
            )
            witness_plan = selftest_witness_plan(
                model,
                requirement,
                selection,
                suffix=suffix,
                index=index,
            )
            witness_fingerprint = structural_sha256(witness_plan)
            witness_fingerprints[(model["model_key"], requirement["provider_id"])] = (
                witness_fingerprint
            )
            provider_evidence.append(
                {
                    "model_key": model["model_key"],
                    "resolved_plan_fingerprint": model[
                        "resolved_plan_fingerprint"
                    ],
                    "plan_hash": model["plan_hash"],
                    "operation_id": requirement["operation_id"],
                    "operation_fingerprint": requirement[
                        "operation_fingerprint"
                    ],
                    "provider_id": requirement["provider_id"],
                    "provider_implementation_fingerprint": requirement[
                        "provider_implementation_fingerprint"
                    ],
                    "provider_execution_contract_fingerprint": requirement[
                        "provider_execution_contract_fingerprint"
                    ],
                    "replay_equivalence": requirement["replay_equivalence"],
                    "required_comparisons": requirement["required_comparisons"],
                    "node_ids": selection["node_ids"],
                    "witness_plan_fingerprint": witness_fingerprint,
                    "witness_plan": witness_plan,
                }
            )
    provider_evidence.sort(
        key=lambda item: (
            item["model_key"],
            item["operation_id"],
            item["provider_id"],
        )
    )
    denominator = {
        "schema_version": {"major": 1, "minor": 0},
        "coverage": coverage,
        "provider_evidence": provider_evidence,
    }
    denominator_path = root / "denominator.json"
    write_structural_json(denominator_path, denominator)
    return denominator, file_sha256(denominator_path), witness_fingerprints


def selftest_witnesses(model_key: str, partition: str, state_kind: str) -> list[dict[str, Any]]:
    base = hashlib.sha256(f"{model_key}/{partition}/{state_kind}".encode()).hexdigest()
    witnesses = []
    for index, suffix in enumerate(("primary", "secondary")):
        node_id = f"node.{model_key}.{suffix}"
        node_digest = hashlib.sha256(f"{base}/{node_id}".encode()).hexdigest()
        witnesses.extend(
            [
                {
                    "kind": "declared_output",
                    "semantic_id": f"value.output.{index}",
                    "node_id": node_id,
                    "resource_id": f"resource.output.{index}",
                    "access": "write",
                    "participant_index": 0,
                    "logical_offset_bytes": 0,
                    "length_bytes": 4,
                    "element_type": "f32",
                    "raw_sha256": node_digest,
                },
                {
                    "kind": "state_effect",
                    "semantic_id": f"state.cache.{index}",
                    "node_id": node_id,
                    "resource_id": f"resource.state.{index}",
                    "access": "read_write",
                    "participant_index": 0,
                    "logical_offset_bytes": 0,
                    "length_bytes": 4,
                    "element_type": "f32",
                    "raw_sha256": hashlib.sha256(
                        (node_digest + "/state").encode()
                    ).hexdigest(),
                },
            ]
        )
    return sorted(witnesses, key=witness_topology)


def make_selftest_execution(
    model_key: str,
    partition: str,
    state_kind: str,
    execution_id: str,
    mode: str,
    participant_count: int,
) -> dict[str, Any]:
    node_ids = [
        f"node.{model_key}.primary",
        f"node.{model_key}.secondary",
    ]
    witnesses = selftest_witnesses(model_key, partition, state_kind)
    if participant_count > 1:
        expanded = []
        for participant_index in range(participant_count):
            for witness in witnesses:
                item = copy.deepcopy(witness)
                item["participant_index"] = participant_index
                expanded.append(item)
        witnesses = sorted(expanded, key=witness_topology)
    replay = mode == "replay"
    return {
        "execution_id": execution_id,
        "mode": mode,
        "restore_sha256": "1" * 64,
        "submission_fingerprint": hashlib.sha256(
            f"{model_key}/{partition}/{execution_id}/submission".encode()
        ).hexdigest(),
        "receipt_fingerprint": hashlib.sha256(
            f"{model_key}/{partition}/{execution_id}/receipt".encode()
        ).hexdigest(),
        "attribution": {
            "batch_identity_fingerprint": "2" * 64,
            "submission_fingerprint": hashlib.sha256(
                f"{model_key}/{partition}/{execution_id}/submission".encode()
            ).hexdigest(),
            "reusable_executable_fingerprint": "3" * 64 if replay else None,
            "commands": [
                {
                    "command_index": command_index,
                    "node_id": node_id,
                    "command_phase": "compute",
                    "native_op_id": "native.test",
                    "execution_path": "replayed" if replay else "eager",
                    "batching_form": "scalar",
                    "participant_count": participant_count,
                    "token_count": participant_count,
                    "compute_dispatch_count": 1,
                    "transfer_command_count": 0,
                    "reusable_graph_node_count": 1 if replay else None,
                }
                for command_index, node_id in enumerate(node_ids)
            ],
        },
        "witnesses": witnesses,
    }


def make_selftest_case(
    model: dict[str, Any],
    phase: str,
    partition: str,
    poison: str,
    state_kind: str,
    denominator_fingerprint: str,
    binary_sha256: str,
    device_fingerprint: str,
    witness_fingerprints: dict[tuple[str, str], str],
) -> dict[str, Any]:
    model_key = model["model_key"]
    if partition in {"single_token", "c1"}:
        participants = 1
        immediate = [1]
        starts = [0 if phase == "prefill" else 8]
    elif partition == "multi_token":
        participants = 1
        immediate = [4]
        starts = [0]
    elif partition == "chunk_boundary":
        participants = 1
        immediate = [2]
        starts = [4]
    elif partition == "multi_participant":
        participants = 2
        immediate = [1, 1]
        starts = [8, 13]
    else:
        participants = 32
        immediate = [1] * participants
        starts = list(range(32))
    ends = [start + tokens for start, tokens in zip(starts, immediate)]
    executions = [
        make_selftest_execution(
            model_key,
            partition,
            state_kind,
            f"eager-{index:02}",
            "eager",
            participants,
        )
        for index in range(6)
    ] + [
        make_selftest_execution(
            model_key,
            partition,
            state_kind,
            f"replay-{index:02}",
            "replay",
            participants,
        )
        for index in range(6)
    ]
    executions.sort(key=lambda item: item["execution_id"])
    comparisons = []
    for index in range(5):
        comparisons.extend(
            [
                {
                    "kind": "eager_eager",
                    "ordinal": index,
                    "left_execution_id": f"eager-{index:02}",
                    "right_execution_id": f"eager-{index + 1:02}",
                    "relation": "bitwise_equal",
                    "first_mismatch": None,
                },
                {
                    "kind": "replay_replay",
                    "ordinal": index,
                    "left_execution_id": f"replay-{index:02}",
                    "right_execution_id": f"replay-{index + 1:02}",
                    "relation": "bitwise_equal",
                    "first_mismatch": None,
                },
                {
                    "kind": "eager_replay",
                    "ordinal": index,
                    "left_execution_id": f"eager-{index:02}",
                    "right_execution_id": f"replay-{index:02}",
                    "relation": "bitwise_equal",
                    "first_mismatch": None,
                },
            ]
        )
    comparisons.sort(key=lambda item: (item["kind"], item["ordinal"]))
    return {
        "schema_version": 1,
        "case_id": f"{model_key}.{phase}.{partition}.{state_kind}.{poison}",
        "denominator_fingerprint": denominator_fingerprint,
        "binary_sha256": binary_sha256,
        "device_runtime_implementation_fingerprint": "a" * 64,
        "device_fingerprint": device_fingerprint,
        "model_key": model_key,
        "resolved_plan_fingerprint": model["resolved_plan_fingerprint"],
        "plan_hash": model["plan_hash"],
        "phase": phase,
        "token_shape": {
            "partition": partition,
            "participant_count": participants,
            "immediate_tokens": immediate,
            "source_start_tokens": starts,
            "source_end_tokens": ends,
        },
        "dtype": "f16",
        "quantization": "gptq_int4",
        "initialization": {
            "input_sha256": "4" * 64,
            "rng_sha256": "5" * 64,
            "initial_state_kind": state_kind,
            "initial_state_sha256": "6" * 64,
            "workspace_poison": poison,
        },
        "coverage_targets": [
            {
                "operation_id": "operation.test",
                "operation_version": {"major": 1, "minor": 0},
                "operation_fingerprint": "c" * 64,
                "provider_id": "provider.cuda.test",
                "provider_version": {"major": 1, "minor": 0},
                "provider_implementation_fingerprint": "d" * 64,
                "provider_execution_contract_fingerprint": "e" * 64,
                "replay_equivalence": "bitwise_eager_equivalent",
                "witness_plan_fingerprint": witness_fingerprints[
                    (model_key, "provider.cuda.test")
                ],
                "node_ids": [f"node.{model_key}.primary"],
            },
            {
                "operation_id": "operation.test.secondary",
                "operation_version": {"major": 1, "minor": 0},
                "operation_fingerprint": "1" * 64,
                "provider_id": "provider.cuda.test.secondary",
                "provider_version": {"major": 1, "minor": 0},
                "provider_implementation_fingerprint": "2" * 64,
                "provider_execution_contract_fingerprint": "3" * 64,
                "replay_equivalence": "bitwise_eager_equivalent",
                "witness_plan_fingerprint": witness_fingerprints[
                    (model_key, "provider.cuda.test.secondary")
                ],
                "node_ids": [f"node.{model_key}.secondary"],
            },
        ],
        "executions": executions,
        "comparisons": comparisons,
        "first_mismatch": None,
    }


def selftest_file_ref(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def make_selftest_hardware_probe(
    root: Path,
    expected_source: dict[str, str],
) -> tuple[dict[str, Any], str]:
    probe_root = root / "hardware-probe"
    outputs = {
        "host": "Linux cuda-host 6.8.0 x86_64\n",
        "gpu": "NVIDIA GeForce RTX 4090, 24564, 555.42\n",
        "toolchain": "Cuda compilation tools, release 12.4, V12.4.99\n",
        "memory": "Mem: 68719476736 0 0 0 0 0\n",
        "cpu": json.dumps(
            {
                "lscpu": [
                    {"field": "CPU(s):", "data": "32"},
                    {"field": "Model name:", "data": "Test CPU"},
                ]
            }
        ),
    }
    errors = {kind: "" for kind in outputs}
    commands = []
    for kind, argv in hardware_probe.PROBE_ARGV["cuda"].items():
        stdout_path = probe_root / "raw" / f"{kind}.stdout.txt"
        stderr_path = probe_root / "raw" / f"{kind}.stderr.txt"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(outputs[kind], encoding="utf-8")
        stderr_path.write_text(errors[kind], encoding="utf-8")
        commands.append(
            {
                "kind": kind,
                "argv": argv,
                "returncode": 0,
                "started_at": "2026-07-27T00:00:00Z",
                "finished_at": "2026-07-27T00:00:01Z",
                "duration_sec": 1.0,
                "stdout": stdout_path.relative_to(probe_root).as_posix(),
                "stdout_sha256": file_sha256(stdout_path),
                "stderr": stderr_path.relative_to(probe_root).as_posix(),
                "stderr_sha256": file_sha256(stderr_path),
            }
        )
    normalized = hardware_probe.normalized_from_outputs(
        "cuda",
        "cuda-g0-1x-rtx4090",
        outputs,
        errors,
    )
    fingerprint = hardware_probe.canonical_sha(normalized)
    probe = {
        "schema_version": 1,
        "source_git_sha": expected_source["git_sha"],
        "source_tree_sha": expected_source["git_tree_sha"],
        "dirty_status": {"is_dirty": False, "status_short": []},
        "collector": {
            "path": HARDWARE_PROBE_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": file_sha256(HARDWARE_PROBE_PATH),
        },
        "hardware_id": "selftest-cuda-rtx4090",
        "normalized": normalized,
        "fingerprint": fingerprint,
        "commands": commands,
    }
    probe_path = probe_root / "probe.json"
    write_json(probe_path, probe)
    return selftest_file_ref(root, probe_path), fingerprint


def make_selftest_models_lock(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    catalog, catalog_sha = validate_model_catalog()
    lock_models = []
    evidence_models = []
    for index, model_key in enumerate(sorted(PRIMARY_MODELS), 1):
        catalog_row = catalog[model_key]
        files = [
            {
                "path": "config.json",
                "sha256": f"{index}" * 64,
                "size_bytes": 1024 + index,
            },
            {
                "path": "model.safetensors",
                "sha256": f"{index + 3}" * 64,
                "size_bytes": 4096 + index,
            },
        ]
        lane = {
            "catalog_lane_id": PRIMARY_MODEL_LANES[model_key],
            "repo": catalog_row["repo"],
            "revision": catalog_row["revision"]["value"],
            "format": catalog_row["format"],
            "files": files,
        }
        lock_models.append({"key": model_key, "lanes": {"cuda": lane}})
        evidence_models.append(
            {
                "model_key": model_key,
                "source_model_id": lane["repo"],
                "revision": lane["revision"],
                "files": files,
                "config_sha256": files[0]["sha256"],
            }
        )
    lock = {
        "schema_version": 1,
        "catalog_sha256": catalog_sha,
        "models": lock_models,
    }
    lock_path = root / "models.lock.json"
    write_json(lock_path, lock)
    return selftest_file_ref(root, lock_path), evidence_models


def make_selftest_artifact(
    root: Path,
    expected_source: dict[str, str],
) -> None:
    denominator, denominator_fingerprint, witness_fingerprints = (
        make_selftest_denominator(root)
    )
    denominator_ref = selftest_file_ref(root, root / "denominator.json")
    binary_path = root / "binary" / "ferrum"
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    binary_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary_path.chmod(0o755)
    binary_ref = selftest_file_ref(root, binary_path)
    models_lock_ref, locked_models = make_selftest_models_lock(root)
    hardware_ref, device_fingerprint = make_selftest_hardware_probe(
        root, expected_source
    )
    stdout_path = root / "runner.stdout"
    stderr_path = root / "runner.stderr"
    stdout_path.write_text("synthetic determinism collector completed\n", encoding="utf-8")
    stderr_path.write_text("synthetic diagnostic stderr\n", encoding="utf-8")
    case_refs = []
    case_shapes = [
        ("prefill", "single_token"),
        ("prefill", "multi_token"),
        ("prefill", "chunk_boundary"),
        ("decode", "c1"),
        ("decode", "multi_participant"),
        ("decode", "c32"),
    ]
    for model in denominator["coverage"]["models"]:
        for phase, partition in case_shapes:
            for state_kind in ("zero", "nonzero"):
                for poison in ("00", "a5"):
                    case = make_selftest_case(
                        model,
                        phase,
                        partition,
                        poison,
                        state_kind,
                        denominator_fingerprint,
                        binary_ref["sha256"],
                        device_fingerprint,
                        witness_fingerprints,
                    )
                    relative = f"cases/{case['case_id']}.json"
                    path = root / relative
                    write_json(path, case)
                    case_refs.append(selftest_file_ref(root, path))
    case_refs.sort(key=lambda item: item["path"])
    models = []
    locked_models_by_key = {
        model["model_key"]: model for model in locked_models
    }
    for model in denominator["coverage"]["models"]:
        locked = locked_models_by_key[model["model_key"]]
        models.append(
            {
                "model_key": model["model_key"],
                "source_model_id": locked["source_model_id"],
                "revision": locked["revision"],
                "files": locked["files"],
                "config_sha256": locked["config_sha256"],
                "external_metadata_id": model["external_metadata_id"],
                "resolved_plan_fingerprint": model["resolved_plan_fingerprint"],
                "plan_hash": model["plan_hash"],
            }
        )
    runner_command = [
        "/workspace/ferrum/target/release/ferrum",
        "vnext-determinism",
        "--models-lock",
        "/workspace/artifacts/models.lock.json",
        "--artifact-root",
        "/workspace/artifacts",
        "--model",
        "m1-qwen35-4b=/workspace/models/m1",
        "--model",
        "m2-qwen35-35b-a3b=/workspace/models/m2",
        "--model",
        "m3-qwen3-30b-a3b=/workspace/models/m3",
    ]
    started_at = "2026-07-27T00:00:00Z"
    finished_at = "2026-07-27T00:01:00Z"
    receipt = {
        "schema": "ferrum.bounded-command-receipt.v1",
        "command": runner_command,
        "cwd": "/workspace/ferrum",
        "pid": 4242,
        "pgid": 4242,
        "limits": {
            "wall_timeout_seconds": 3600.0,
            "max_processes": 16,
            "max_group_threads": 256,
            "max_per_process_threads": 128,
            "sample_interval_seconds": 0.05,
            "max_sampling_errors": 3,
            "term_grace_seconds": 2.0,
        },
        "peaks": {
            "processes": 1,
            "group_threads": 32,
            "per_process_threads": 32,
            "per_process_threads_pid": 4242,
        },
        "started_at": started_at,
        "ended_at": finished_at,
        "duration_seconds": 60.0,
        "reason": "command_completed",
        "rc": 0,
        "status": "pass",
        "successful_samples": 20,
        "sampling_error_count": 0,
        "sampling_errors": [],
        "violation": None,
        "termination": {"signals": [], "errors": []},
        "cleanup": {"process_group_gone": True},
        "stdout": {
            "path": "/workspace/artifacts/runner.stdout",
            "sha256": file_sha256(stdout_path),
            "size_bytes": stdout_path.stat().st_size,
        },
        "stderr": {
            "path": "/workspace/artifacts/runner.stderr",
            "sha256": file_sha256(stderr_path),
            "size_bytes": stderr_path.stat().st_size,
        },
    }
    receipt_path = root / "runner.receipt.json"
    write_json(receipt_path, receipt)
    evidence = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "backend": "cuda",
        "source": {
            "git_sha": expected_source["git_sha"],
            "git_tree_sha": expected_source["git_tree_sha"],
            "dirty_status": [],
            "build_command": [
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
            "binary_path": "/workspace/ferrum/target/release/ferrum",
            "binary": binary_ref,
        },
        "hardware": {
            "probe": hardware_ref,
            "fingerprint": device_fingerprint,
        },
        "models_lock": models_lock_ref,
        "denominator": {
            **denominator_ref,
            "fingerprint": denominator_fingerprint,
        },
        "models": models,
        "runner": {
            "command": runner_command,
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0,
            "receipt": {
                **selftest_file_ref(root, receipt_path),
            },
            "stdout": selftest_file_ref(root, stdout_path),
            "stderr": selftest_file_ref(root, stderr_path),
        },
        "cases": case_refs,
    }
    write_json(root / "evidence.json", evidence)


def refresh_case_ref(root: Path, case_path: Path) -> None:
    evidence = read_json(root / "evidence.json")
    relative = case_path.relative_to(root).as_posix()
    for ref in evidence["cases"]:
        if ref["path"] == relative:
            ref["sha256"] = file_sha256(case_path)
            ref["size_bytes"] = case_path.stat().st_size
            break
    write_json(root / "evidence.json", evidence)


def run_self_test() -> None:
    expected_source = {
        "git_sha": "1" * 40,
        "git_tree_sha": "2" * 40,
    }
    with tempfile.TemporaryDirectory(prefix="ferrum-vnext-cuda-determinism-") as temporary:
        base = Path(temporary) / "base"
        base.mkdir()
        make_selftest_artifact(base, expected_source)
        summary = validate_artifact(base, expected_source)
        require(summary["case_count"] == 72, "self-test case denominator drifted")
        require(summary["comparison_count"] == 1080, "self-test comparison denominator drifted")

        mutations: list[tuple[str, Callable[[Path], None], str]] = []

        def mutate_case(
            relative: str,
            mutation: Callable[[dict[str, Any]], None],
        ) -> Callable[[Path], None]:
            def apply(root: Path) -> None:
                path = root / relative
                value = read_json(path)
                mutation(value)
                write_json(path, value)
                refresh_case_ref(root, path)

            return apply

        first_case = "cases/m1-qwen35-4b.prefill.single_token.zero.00.json"
        replay_case = "cases/m1-qwen35-4b.decode.c1.zero.00.json"

        def mutate_receipt_sampling_error(root: Path) -> None:
            receipt_path = root / "runner.receipt.json"
            receipt = read_json(receipt_path)
            receipt["sampling_error_count"] = 1
            receipt["sampling_errors"] = [
                {
                    "at": "2026-07-27T00:00:30Z",
                    "type": "Synthetic",
                    "error": "sample failed",
                }
            ]
            write_json(receipt_path, receipt)
            evidence = read_json(root / "evidence.json")
            evidence["runner"]["receipt"]["sha256"] = file_sha256(receipt_path)
            evidence["runner"]["receipt"]["size_bytes"] = receipt_path.stat().st_size
            write_json(root / "evidence.json", evidence)

        def mutate_binary_bytes(root: Path) -> None:
            with (root / "binary" / "ferrum").open("a", encoding="utf-8") as handle:
                handle.write("# tampered\n")

        def mutate_models_lock_catalog(root: Path) -> None:
            lock_path = root / "models.lock.json"
            lock = read_json(lock_path)
            lock["catalog_sha256"] = "0" * 64
            write_json(lock_path, lock)
            evidence = read_json(root / "evidence.json")
            evidence["models_lock"] = selftest_file_ref(root, lock_path)
            write_json(root / "evidence.json", evidence)

        def mutate_hardware_raw_output(root: Path) -> None:
            probe_path = root / "hardware-probe" / "probe.json"
            probe = read_json(probe_path)
            command = next(
                item for item in probe["commands"] if item["kind"] == "gpu"
            )
            stdout_path = probe_path.parent / command["stdout"]
            stdout_path.write_text(
                "NVIDIA GeForce RTX 4090 Selftest-Tampered, 24564, 555.42\n",
                encoding="utf-8",
            )
            command["stdout_sha256"] = file_sha256(stdout_path)
            write_json(probe_path, probe)
            evidence = read_json(root / "evidence.json")
            evidence["hardware"]["probe"] = selftest_file_ref(root, probe_path)
            write_json(root / "evidence.json", evidence)

        def mutate_typed_witness_denominator(root: Path) -> None:
            denominator_path = root / "denominator.json"
            denominator = read_json(denominator_path)
            denominator["provider_evidence"][0]["witness_plan"]["witnesses"][0][
                "location"
            ]["declared_length_bytes"] = 8
            write_structural_json(denominator_path, denominator)
            evidence = read_json(root / "evidence.json")
            ref = selftest_file_ref(root, denominator_path)
            evidence["denominator"] = {**ref, "fingerprint": ref["sha256"]}
            write_json(root / "evidence.json", evidence)

        mutations.extend(
            [
                (
                    "binary-tamper",
                    mutate_binary_bytes,
                    "source.binary.size_bytes differs",
                ),
                (
                    "models-lock-stale",
                    mutate_models_lock_catalog,
                    "models.lock is stale",
                ),
                (
                    "hardware-raw-tamper",
                    mutate_hardware_raw_output,
                    "normalized facts are not derived",
                ),
                (
                    "typed-witness-denominator-tamper",
                    mutate_typed_witness_denominator,
                    "not derived from the typed witness plan",
                ),
                (
                    "runner-sampling-error",
                    mutate_receipt_sampling_error,
                    "contains sampling errors",
                ),
                (
                    "stale-runtime",
                    mutate_case(
                        first_case,
                        lambda value: value.update(
                            {"device_runtime_implementation_fingerprint": "0" * 64}
                        ),
                    ),
                    "runtime fingerprint is stale",
                ),
                (
                    "missing-node-witness",
                    mutate_case(
                        first_case,
                        lambda value: [
                            execution.update(
                                {
                                    "witnesses": [
                                        item
                                        for item in execution["witnesses"]
                                        if item["kind"] != "declared_output"
                                    ]
                                }
                            )
                            for execution in value["executions"]
                        ],
                    ),
                    "declared node output",
                ),
                (
                    "missing-comparison-kind",
                    mutate_case(
                        first_case,
                        lambda value: value.update(
                            {
                                "comparisons": [
                                    item
                                    for item in value["comparisons"]
                                    if item["kind"] != "replay_replay"
                                ]
                            }
                        ),
                    ),
                    "missing a required comparison",
                ),
                (
                    "replay-fallback",
                    mutate_case(
                        replay_case,
                        lambda value: value["executions"][-1]["attribution"]["commands"][0].update(
                            {
                                "execution_path": "eager",
                                "reusable_graph_node_count": None,
                            }
                        ),
                    ),
                    "eager fallback",
                ),
                (
                    "tolerance-field",
                    mutate_case(
                        first_case,
                        lambda value: value["comparisons"][0].update(
                            {"absolute_tolerance": 0.001}
                        ),
                    ),
                    "unknown fields",
                ),
                (
                    "raw-mismatch",
                    mutate_case(
                        first_case,
                        lambda value: value["executions"][1]["witnesses"][0].update(
                            {"raw_sha256": "0" * 64}
                        ),
                    ),
                    "raw witness mismatch",
                ),
                (
                    "restore-drift",
                    mutate_case(
                        first_case,
                        lambda value: value["executions"][1].update(
                            {"restore_sha256": "0" * 64}
                        ),
                    ),
                    "do not restore the same input/rng/initial-state bytes",
                ),
                (
                    "cross-model",
                    mutate_case(
                        first_case,
                        lambda value: value.update({"model_key": "m2-qwen35-35b-a3b"}),
                    ),
                    "resolved model plan",
                ),
            ]
        )

        def remove_cases(
            predicate: Callable[[str], bool],
        ) -> Callable[[Path], None]:
            def apply(root: Path) -> None:
                evidence = read_json(root / "evidence.json")
                removed = [item for item in evidence["cases"] if predicate(item["path"])]
                evidence["cases"] = [item for item in evidence["cases"] if not predicate(item["path"])]
                for item in removed:
                    (root / item["path"]).unlink()
                write_json(root / "evidence.json", evidence)

            return apply

        def mutate_all_cases(
            mutation: Callable[[dict[str, Any]], None],
        ) -> Callable[[Path], None]:
            def apply(root: Path) -> None:
                evidence = read_json(root / "evidence.json")
                for ref in evidence["cases"]:
                    path = root / ref["path"]
                    value = read_json(path)
                    mutation(value)
                    write_json(path, value)
                    ref["sha256"] = file_sha256(path)
                    ref["size_bytes"] = path.stat().st_size
                write_json(root / "evidence.json", evidence)

            return apply

        def drop_secondary_provider(value: dict[str, Any]) -> None:
            secondary_nodes = set(value["coverage_targets"][1]["node_ids"])
            value["coverage_targets"] = value["coverage_targets"][:1]
            for execution in value["executions"]:
                commands = [
                    command
                    for command in execution["attribution"]["commands"]
                    if command["node_id"] not in secondary_nodes
                ]
                for command_index, command in enumerate(commands):
                    command["command_index"] = command_index
                execution["attribution"]["commands"] = commands
                execution["witnesses"] = [
                    witness
                    for witness in execution["witnesses"]
                    if witness["node_id"] not in secondary_nodes
                ]

        mutations.extend(
            [
                (
                    "missing-c32",
                    remove_cases(lambda path: ".decode.c32." in path),
                    "shape partition coverage",
                ),
                (
                    "missing-provider",
                    mutate_all_cases(drop_secondary_provider),
                    "lacks exact node proof coverage",
                ),
                (
                    "single-poison",
                    mutate_all_cases(
                        lambda value: value["initialization"].update(
                            {"workspace_poison": "00"}
                        )
                    ),
                    "lacks both workspace poison patterns",
                ),
                (
                    "zero-state-only",
                    mutate_all_cases(
                        lambda value: value["initialization"].update(
                            {"initial_state_kind": "zero"}
                        )
                    ),
                    "lacks zero/nonzero state-effect evidence",
                ),
            ]
        )

        for name, mutation, marker in mutations:
            case = Path(temporary) / name
            shutil.copytree(base, case)
            mutation(case)
            try:
                validate_artifact(case, expected_source)
            except DeterminismArtifactError as error:
                require(
                    marker.lower() in str(error).lower(),
                    f"mutation {name} rejected for the wrong reason: {error}",
                )
            else:
                raise AssertionError(f"mutation {name} unexpectedly passed")
    print("RUNTIME VNEXT CUDA DETERMINISM SELF-TEST PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", nargs="?", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.artifact_root is None or args.out is None:
        raise SystemExit("artifact_root and --out are required")
    out_dir = args.out.resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        require(not any(out_dir.iterdir()), "--out must be empty")
        summary = validate_artifact(args.artifact_root, current_git_state())
        manifest = validation_manifest(args.artifact_root, out_dir, summary)
        write_json(out_dir / "manifest.json", manifest, exclusive=True)
        print(manifest["pass_line"])
        return 0
    except (DeterminismArtifactError, OSError) as error:
        write_rejection_manifest(args.artifact_root, out_dir, error)
        print(
            f"FERRUM RUNTIME VNEXT CUDA DETERMINISM REJECT: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
