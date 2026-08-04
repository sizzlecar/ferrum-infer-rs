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
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:
    import runtime_vnext_hardware_probe as hardware_probe
except ModuleNotFoundError:
    from scripts.release import runtime_vnext_hardware_probe as hardware_probe

try:
    import runtime_vnext_baseline_scenarios as baseline_scenarios
except ModuleNotFoundError:
    from scripts.release import runtime_vnext_baseline_scenarios as baseline_scenarios


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_CATALOG_PATH = REPO_ROOT / "scripts/release/configs/runtime_vnext_models.json"
HARDWARE_PROBE_PATH = REPO_ROOT / "scripts/release/runtime_vnext_hardware_probe.py"
EVIDENCE_COLLECTOR_PATH = (
    REPO_ROOT / "scripts/release/runtime_vnext_cuda_determinism_collect.py"
)
ARTIFACT_TYPE = "runtime_vnext_cuda_determinism_evidence"
VALIDATOR_ARTIFACT_TYPE = "runtime_vnext_cuda_determinism_validation"
PRIMARY_MODEL_LANES = {
    "m1-qwen35-4b": "M1-CUDA",
    "m2-qwen35-35b-a3b": "M2-CUDA",
    "m3-qwen3-30b-a3b": "M3-CUDA",
}
PRIMARY_MODELS = set(PRIMARY_MODEL_LANES)
RELEASE_SCOPE = "release-full"
M1_S2_FOCUSED_SCOPE = "m1-s2-focused"
M1_MODEL = "m1-qwen35-4b"
RELEASE_PARTITIONS = {
    ("prefill", "single_token"),
    ("prefill", "multi_token"),
    ("prefill", "chunk_boundary"),
    ("decode", "c1"),
    ("decode", "multi_participant"),
    ("decode", "c32"),
}
M1_S2_FOCUSED_PARTITIONS = RELEASE_PARTITIONS - {("decode", "c32")}
SCOPE_CONTRACTS = {
    RELEASE_SCOPE: {
        "models": PRIMARY_MODELS,
        "partitions": RELEASE_PARTITIONS,
        "pass_prefix": "FERRUM RUNTIME VNEXT CUDA DETERMINISM PASS",
        "lane": "runtime-vnext-cuda-determinism",
    },
    M1_S2_FOCUSED_SCOPE: {
        "models": {M1_MODEL},
        "partitions": M1_S2_FOCUSED_PARTITIONS,
        "pass_prefix": "FERRUM RUNTIME VNEXT M1 S2 CUDA DETERMINISM FOCUSED PASS",
        "lane": "runtime-vnext-m1-s2-cuda-determinism-focused",
    },
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
MAX_MODEL_VERIFICATION_JSON_BYTES = 64 * 1024 * 1024
MAX_DENOMINATOR_JSON_BYTES = 128 * 1024 * 1024
MAX_CASE_JSON_BYTES = 32 * 1024 * 1024
MAX_LOG_BYTES = 128 * 1024 * 1024
MAX_BINARY_BYTES = 1024 * 1024 * 1024
BUILD_PROVENANCE_ROOT = Path("build-provenance")
CANDIDATE_BUILD_RECEIPT_PATH = (
    BUILD_PROVENANCE_ROOT / "build/candidate/candidate-build-receipt.json"
)
CANDIDATE_BUILD_BINARY_PATH = BUILD_PROVENANCE_ROOT / "build/candidate/ferrum"
NATIVE_OPERATOR_SET_LOCK_PATH = (
    BUILD_PROVENANCE_ROOT
    / baseline_scenarios.CANDIDATE_NATIVE_OPERATOR_SET_LOCK_REL
)
DETERMINISM_WORKER_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "8",
    "TOKIO_WORKER_THREADS": "8",
    "VECLIB_MAXIMUM_THREADS": "1",
}
DETERMINISM_FIXED_ENVIRONMENT = {
    **DETERMINISM_WORKER_ENVIRONMENT,
    "CUDA_VISIBLE_DEVICES": "0",
}
DETERMINISM_ENVIRONMENT_PASSTHROUGH = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_REQUIRE_CUDA",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
        "TEMP",
        "TMP",
        "TMPDIR",
    }
)

ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "backend",
        "scope",
        "source",
        "hardware",
        "models_lock",
        "model_verification",
        "collector",
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
        "binary_path",
        "binary",
        "candidate_build_receipt",
        "native_operator_set_lock",
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
MODEL_VERIFICATION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "scope",
        "source_git_sha",
        "source_tree_sha",
        "collector",
        "verified_at",
        "models",
    }
)
MODEL_VERIFICATION_COLLECTOR_FIELDS = frozenset({"path", "sha256"})
MODEL_VERIFICATION_MODEL_FIELDS = frozenset(
    {"model_key", "model_dir", "files", "consumed_files"}
)
RUNNER_FIELDS = frozenset(
    {
        "command",
        "environment",
        "repository_root",
        "started_at",
        "finished_at",
        "exit_code",
        "receipt",
        "stdout",
        "stderr",
    }
)
COLLECTOR_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "status",
        "backend",
        "scope",
        "models_lock",
        "hardware_probe",
        "device_fingerprint",
        "binary",
        "denominator",
        "models",
        "cases",
        "case_count",
        "execution_count",
        "comparison_count",
        "pass_line",
    }
)
COLLECTOR_MODEL_FIELDS = frozenset(
    {
        "model_key",
        "model_dir",
        "resolved_plan_fingerprint",
        "plan_hash",
        "dtype",
        "quantization",
        "case_count",
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
        "provider_coverage",
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
        "compute_path_requirement",
        "reusable_program_fingerprint",
        "declared_eager_boundary_node_ids",
        "restore_sha256",
        "initialization_identity",
        "submission_fingerprint",
        "receipt_fingerprint",
        "attribution",
        "witnesses",
    }
)
EXECUTION_INITIALIZATION_IDENTITY_FIELDS = frozenset(
    {"input_sha256", "rng_sha256", "initial_state_sha256"}
)
ATTRIBUTION_FIELDS = frozenset(
    {
        "batch_identity_fingerprint",
        "submission_fingerprint",
        "physical_commands",
        "replayed_segments",
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
REPLAYED_SEGMENT_FIELDS = frozenset(
    {
        "physical_command_index",
        "reusable_program_fingerprint",
        "reusable_executable_fingerprint",
        "logical_commands",
    }
)
REPLAYED_LOGICAL_COMMAND_FIELDS = frozenset(
    {
        "logical_command_ordinal",
        "node_id",
        "native_op_id",
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


def scope_contract(scope: str) -> dict[str, Any]:
    contract = SCOPE_CONTRACTS.get(scope)
    require(contract is not None, f"unknown determinism scope: {scope}")
    return contract


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
    allow_empty: bool = False,
) -> tuple[dict[str, Any], Path]:
    ref = exact_object(value, FILE_REF_FIELDS, label)
    path = safe_artifact_file(root, ref["path"], label)
    size = integer(
        ref["size_bytes"],
        f"{label}.size_bytes",
        minimum=0 if allow_empty else 1,
    )
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
    canonical_order: bool = True,
) -> list[str]:
    require(isinstance(value, list), f"{label} must be a list")
    if nonempty:
        require(bool(value), f"{label} must not be empty")
    result = [text(item, f"{label}[{index}]", portable=portable) for index, item in enumerate(value)]
    if canonical_order:
        require(result == sorted(set(result)), f"{label} must be sorted and unique")
    else:
        require(len(result) == len(set(result)), f"{label} must be unique")
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
    repository_root: Path,
    recorded_artifact_root: Path,
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
    receipt_cwd = Path(text(receipt["cwd"], "runner.receipt.cwd"))
    require(
        receipt_cwd.is_absolute() and receipt_cwd == repository_root,
        "runner receipt cwd differs from the recorded repository root",
    )
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
    require(
        (max_processes, max_group_threads, max_per_process_threads)
        == (16, 128, 64),
        "runner receipt must use the fixed 16/128/64 process-thread bounds",
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
        output_path = Path(text(output["path"], f"runner.receipt.{name}.path"))
        require(
            output_path
            == recorded_artifact_root / "runner" / f"{name}.log",
            f"runner receipt {name} path differs from the canonical artifact path",
        )
        require(
            output["sha256"] == expected["sha256"]
            and output["size_bytes"] == expected["size_bytes"],
            f"runner receipt {name} identity differs from copied evidence",
        )


def validate_runner_environment(value: Any) -> dict[str, str]:
    require(isinstance(value, dict), "evidence.runner.environment must be an object")
    allowed = set(DETERMINISM_FIXED_ENVIRONMENT) | set(
        DETERMINISM_ENVIRONMENT_PASSTHROUGH
    )
    require(
        set(value) <= allowed and "PATH" in value,
        "runner environment contains an undeclared variable or lacks PATH",
    )
    environment: dict[str, str] = {}
    for key in sorted(value):
        require(
            isinstance(key, str)
            and key
            and isinstance(value[key], str)
            and len(value[key]) <= 16_384,
            f"runner environment value is invalid for {key!r}",
        )
        environment[key] = value[key]
    require(
        all(environment.get(key) == expected for key, expected in DETERMINISM_FIXED_ENVIRONMENT.items()),
        "runner environment does not enforce the fixed CUDA/worker limits",
    )
    return environment


def canonical_determinism_environment(source: Mapping[str, str]) -> dict[str, str]:
    environment = {
        key: source[key]
        for key in sorted(DETERMINISM_ENVIRONMENT_PASSTHROUGH)
        if key in source
    }
    environment.update(DETERMINISM_FIXED_ENVIRONMENT)
    return validate_runner_environment(environment)


def exact_collector_pass_line(scope: str, artifact_root: str) -> str:
    prefix = (
        "FERRUM VNEXT M1 S2 FOCUSED DETERMINISM COLLECTOR PASS"
        if scope == M1_S2_FOCUSED_SCOPE
        else "FERRUM VNEXT DETERMINISM COLLECTOR PASS"
    )
    return f"{prefix}: {artifact_root}"


def validate_collector_manifest(
    root: Path,
    value: Any,
    *,
    scope: str,
    expected_models: set[str],
    runner_command: list[str],
    runner_stdout_path: Path,
    binary_ref: dict[str, Any],
    models_lock_ref: dict[str, Any],
    hardware_probe_ref: dict[str, Any],
    device_fingerprint: str,
    denominator_ref: dict[str, Any],
    denominator: dict[str, Any],
    model_directories: dict[str, str],
    case_refs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    collector_ref, collector_path = validate_file_ref(
        root,
        value,
        "evidence.collector",
        max_size_bytes=MAX_ROOT_JSON_BYTES,
    )
    require(
        collector_ref["path"] == "collector.json",
        "evidence.collector must reference collector.json",
    )
    collector = exact_object(
        read_json(collector_path, max_bytes=MAX_ROOT_JSON_BYTES),
        COLLECTOR_FIELDS,
        "collector",
    )
    require(
        collector["schema_version"] == 1
        and collector["artifact_type"] == "runtime_vnext_cuda_determinism_collector"
        and collector["status"] == "pass"
        and collector["backend"] == "cuda"
        and collector["scope"] == scope,
        "collector identity, status, backend, or scope is invalid",
    )
    expected_pass = exact_collector_pass_line(scope, runner_command[5])
    require(collector["pass_line"] == expected_pass, "collector PASS line is invalid")
    require(
        expected_pass
        in runner_stdout_path.read_text(encoding="utf-8", errors="strict").splitlines(),
        "runner stdout lacks the exact Rust collector PASS line",
    )
    require(
        collector["models_lock"] == models_lock_ref,
        "collector models.lock reference differs from evidence",
    )
    require(
        collector["hardware_probe"] == hardware_probe_ref
        and collector["device_fingerprint"] == device_fingerprint,
        "collector hardware identity differs from evidence",
    )
    require(
        collector["denominator"] == denominator_ref,
        "collector denominator reference differs from evidence",
    )
    collector_binary = exact_object(
        collector["binary"], FILE_REF_FIELDS, "collector.binary"
    )
    require(
        collector_binary
        == {
            "path": runner_command[0],
            "sha256": binary_ref["sha256"],
            "size_bytes": binary_ref["size_bytes"],
        },
        "collector binary identity differs from the executed release binary",
    )

    expected_case_count = len(expected_models) * len(scope_contract(scope)["partitions"]) * 4
    require(
        integer(collector["case_count"], "collector.case_count", minimum=1)
        == expected_case_count
        and collector["cases"] == case_refs,
        "collector case references or exact denominator differ from evidence",
    )
    integer(collector["execution_count"], "collector.execution_count", minimum=1)
    integer(collector["comparison_count"], "collector.comparison_count", minimum=1)

    rows = collector["models"]
    require(isinstance(rows, list), "collector.models must be a list")
    cases_per_model = len(scope_contract(scope)["partitions"]) * 4
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        label = f"collector.models[{index}]"
        row = exact_object(raw, COLLECTOR_MODEL_FIELDS, label)
        model_key = text(row["model_key"], f"{label}.model_key", portable=True)
        require(
            model_key in expected_models and model_key not in indexed,
            f"{label}.model_key is invalid or duplicated",
        )
        planned = denominator["models"][model_key]
        require(
            row["model_dir"] == model_directories[model_key]
            and row["resolved_plan_fingerprint"]
            == planned["resolved_plan_fingerprint"]
            and row["plan_hash"] == planned["plan_hash"]
            and row["case_count"] == cases_per_model,
            f"{label} differs from the verified model binding or denominator",
        )
        text(row["dtype"], f"{label}.dtype", portable=True)
        text(row["quantization"], f"{label}.quantization", portable=True)
        indexed[model_key] = row
    require(
        list(indexed) == sorted(expected_models),
        "collector model set or ordering differs from the selected scope",
    )
    return collector, indexed


def validate_coverage(
    value: dict[str, Any], expected_models: set[str], *, allow_unselected: bool
) -> dict[str, Any]:
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
    require(
        len(models_raw) == len(expected_models),
        "coverage model cardinality differs from the selected scope",
    )
    models: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(models_raw):
        model = exact_object(raw, COVERAGE_MODEL_FIELDS, f"coverage.models[{index}]")
        key = text(model["model_key"], f"coverage.models[{index}].model_key", portable=True)
        require(
            key in expected_models and key not in models,
            "coverage model key is invalid or duplicated for the selected scope",
        )
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
            canonical_order=False,
        )
        models[key] = model
    require(list(models) == sorted(models), "coverage.models must use canonical model-key order")
    require(set(models) == expected_models, "coverage model set differs from the selected scope")

    requirements_raw = coverage["provider_requirements"]
    require(
        isinstance(requirements_raw, list) and bool(requirements_raw),
        "coverage.provider_requirements must be non-empty",
    )
    requirements: dict[tuple[str, str], dict[str, Any]] = {}
    previous_key: tuple[str, str] | None = None
    model_nodes: dict[str, set[str]] = {key: set() for key in expected_models}
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
            isinstance(selections_raw, list)
            and (bool(selections_raw) or allow_unselected),
            f"{label}.model_selections cannot be empty for full catalog coverage",
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
                canonical_order=False,
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
    elif kind == "immediate_token_span":
        exact_object(extent, VALUE_EXTENT_TOKEN_FIELDS, f"{label}.extent")
        bytes_per_token = integer(
            extent["bytes_per_token"], f"{label}.extent.bytes_per_token", minimum=1
        )
        maximum_tokens = integer(
            extent["maximum_tokens"], f"{label}.extent.maximum_tokens", minimum=1
        )
        require(
            declared % bytes_per_token == 0,
            f"{label}.extent does not divide the declared location into whole tokens",
        )
        require(
            bytes_per_token * maximum_tokens <= (1 << 64) - 1,
            f"{label}.extent maximum token span overflows u64",
        )
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
        require(
            maximum_storage >= declared,
            f"{label}.extent is smaller than its declared location",
        )
    else:
        raise DeterminismArtifactError(f"{label}.extent.kind is invalid")
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
    node_ids = validate_string_list(
        plan["node_ids"],
        f"{label}.node_ids",
        portable=True,
        canonical_order=False,
    )
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


def validate_denominator(
    value: dict[str, Any], expected_models: set[str], scope: str
) -> dict[str, Any]:
    denominator = exact_object(value, DENOMINATOR_FIELDS, "denominator")
    require(
        validate_version(denominator["schema_version"], "denominator.schema_version")
        == (1, 1),
        "denominator.schema_version must be 1.1",
    )
    expected_provider_coverage = (
        "selected_plan_providers"
        if scope == M1_S2_FOCUSED_SCOPE
        else "all_catalog_providers"
    )
    require(
        denominator["provider_coverage"] == expected_provider_coverage,
        "denominator.provider_coverage differs from the selected scope",
    )
    coverage = validate_coverage(
        denominator["coverage"],
        expected_models,
        allow_unselected=expected_provider_coverage == "selected_plan_providers",
    )
    evidence_rows = denominator["provider_evidence"]
    require(
        isinstance(evidence_rows, list)
        and 0 < len(evidence_rows) <= len(expected_models) * 512,
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
        "provider_coverage": expected_provider_coverage,
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


def validate_model_verification(
    root: Path,
    value: Any,
    *,
    expected_scope: str,
    expected_models: set[str],
    expected_source: dict[str, Any],
    locked_models: dict[str, dict[str, Any]],
) -> dict[str, str]:
    _, path = validate_file_ref(
        root,
        value,
        "evidence.model_verification",
        max_size_bytes=MAX_MODEL_VERIFICATION_JSON_BYTES,
    )
    document = exact_object(
        read_json(path, max_bytes=MAX_MODEL_VERIFICATION_JSON_BYTES),
        MODEL_VERIFICATION_FIELDS,
        "evidence.model_verification.document",
    )
    require(
        document["schema_version"] == 1
        and document["artifact_type"]
        == "runtime_vnext_cuda_determinism_model_verification",
        "model verification schema or artifact type is invalid",
    )
    require(
        text(document["scope"], "model verification scope") == expected_scope,
        "model verification scope differs from evidence",
    )
    require(
        document["source_git_sha"] == expected_source["git_sha"]
        and document["source_tree_sha"] == expected_source["git_tree_sha"],
        "model verification source identity is stale",
    )
    collector = exact_object(
        document["collector"],
        MODEL_VERIFICATION_COLLECTOR_FIELDS,
        "model verification collector",
    )
    require(
        collector
        == {
            "path": EVIDENCE_COLLECTOR_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": file_sha256(EVIDENCE_COLLECTOR_PATH),
        },
        "model verification collector identity is stale",
    )
    validate_timestamp(document["verified_at"], "model verification verified_at")
    rows = document["models"]
    require(isinstance(rows, list), "model verification models must be a list")
    directories: dict[str, str] = {}
    for index, raw in enumerate(rows):
        label = f"model verification models[{index}]"
        row = exact_object(raw, MODEL_VERIFICATION_MODEL_FIELDS, label)
        model_key = text(row["model_key"], f"{label}.model_key", portable=True)
        require(
            model_key in expected_models and model_key not in directories,
            f"{label}.model_key is invalid or duplicated",
        )
        model_dir = text(row["model_dir"], f"{label}.model_dir")
        require(Path(model_dir).is_absolute(), f"{label}.model_dir must be absolute")
        files = row["files"]
        require(isinstance(files, list) and files, f"{label}.files must be non-empty")
        normalized_files = []
        for file_index, file_raw in enumerate(files):
            file_label = f"{label}.files[{file_index}]"
            model_file = exact_object(file_raw, MODEL_FILE_FIELDS, file_label)
            normalized_files.append(
                {
                    "path": text(model_file["path"], f"{file_label}.path"),
                    "sha256": sha256_text(
                        model_file["sha256"], f"{file_label}.sha256"
                    ),
                    "size_bytes": integer(
                        model_file["size_bytes"],
                        f"{file_label}.size_bytes",
                        minimum=1,
                    ),
                }
            )
        require(
            normalized_files == locked_models[model_key]["files"],
            f"{label}.files differ from models.lock",
        )
        consumed_files = validate_string_list(
            row["consumed_files"],
            f"{label}.consumed_files",
        )
        locked_paths = {item["path"] for item in normalized_files}
        require(
            set(consumed_files) <= locked_paths
            and {"config.json", "tokenizer.json"} <= set(consumed_files)
            and any(path.endswith(".safetensors") for path in consumed_files),
            f"{label}.consumed_files is not a locked production loader closure",
        )
        directories[model_key] = model_dir
    require(
        list(directories) == sorted(expected_models),
        "model verification model set differs from the selected scope",
    )
    return directories


def validate_hardware(
    root: Path,
    value: Any,
    *,
    source: dict[str, Any],
) -> tuple[str, str]:
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
    hardware_id = text(probe["hardware_id"], "hardware probe hardware_id", portable=True)
    return fingerprint, hardware_id


def validate_candidate_build_provenance(
    root: Path,
    source: dict[str, Any],
    *,
    hardware_id: str,
    binary_ref: dict[str, Any],
    allow_internal_fixture: bool,
) -> dict[str, Any]:
    receipt_ref, receipt_path = validate_file_ref(
        root,
        source["candidate_build_receipt"],
        "evidence.source.candidate_build_receipt",
        max_size_bytes=MAX_RECEIPT_JSON_BYTES,
    )
    require(
        receipt_ref["path"] == CANDIDATE_BUILD_RECEIPT_PATH.as_posix(),
        "candidate build receipt is outside the canonical imported layout",
    )
    provenance_root = root / BUILD_PROVENANCE_ROOT
    nested_ref = {
        "kind": "raw-json",
        "path": receipt_path.relative_to(provenance_root).as_posix(),
        "sha256": receipt_ref["sha256"],
    }
    try:
        receipt, _, _, imported_binary_path = (
            baseline_scenarios.validate_candidate_build_receipt(
                provenance_root,
                nested_ref,
                expected={
                    "source_git_sha": source["git_sha"],
                    "source_tree_sha": source["git_tree_sha"],
                    "hardware_id": hardware_id,
                    "backend": "cuda",
                    "binary_sha256": binary_ref["sha256"],
                },
                allow_internal_fixture=allow_internal_fixture,
            )
        )
    except baseline_scenarios.ScenarioError as error:
        raise DeterminismArtifactError(
            f"candidate build receipt is invalid: {error}"
        ) from error
    require(
        imported_binary_path == root / CANDIDATE_BUILD_BINARY_PATH
        and imported_binary_path.stat().st_size == binary_ref["size_bytes"],
        "candidate build binary differs from the canonical imported binary",
    )

    lock_ref, _ = validate_file_ref(
        root,
        source["native_operator_set_lock"],
        "evidence.source.native_operator_set_lock",
        max_size_bytes=MAX_MODEL_LOCK_JSON_BYTES,
    )
    require(
        lock_ref["path"] == NATIVE_OPERATOR_SET_LOCK_PATH.as_posix(),
        "native operator set lock is outside the canonical imported layout",
    )
    recorded_lock = receipt.get("native_operator_set_lock")
    require(
        isinstance(recorded_lock, dict)
        and lock_ref["sha256"] == recorded_lock.get("sha256")
        and lock_ref["size_bytes"] == recorded_lock.get("size_bytes"),
        "imported native operator set lock differs from the candidate build receipt",
    )
    return receipt


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
    exact_shapes = {
        ("prefill", "single_token"): (1, [1], [0], [1]),
        ("prefill", "multi_token"): (1, [4], [0], [4]),
        ("prefill", "chunk_boundary"): (1, [4], [4], [8]),
        ("decode", "c1"): (1, [1], [8], [9]),
        ("decode", "multi_participant"): (4, [1] * 4, [8] * 4, [9] * 4),
        ("decode", "c32"): (32, [1] * 32, [8] * 32, [9] * 32),
    }
    expected = exact_shapes.get((phase, partition))
    require(expected is not None, f"{label}.partition is invalid for {phase}")
    require(
        (participant_count, immediate, starts, ends) == expected,
        f"{label} differs from the canonical Rust shape fixture",
    )
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
    node_ids = set(
        validate_string_list(
            target["node_ids"],
            f"{label}.node_ids",
            portable=True,
            canonical_order=False,
        )
    )
    require(
        node_ids == set(selection["node_ids"]),
        f"{label}.node_ids differ from the exact selected provider nodes",
    )
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
) -> tuple[dict[str, Any], dict[tuple[Any, ...], str], set[str]]:
    execution = exact_object(value, EXECUTION_FIELDS, label)
    text(execution["execution_id"], f"{label}.execution_id", portable=True)
    mode = text(execution["mode"], f"{label}.mode")
    require(mode in {"eager", "replay"}, f"{label}.mode is invalid")
    compute_path_requirement = text(
        execution["compute_path_requirement"],
        f"{label}.compute_path_requirement",
    )
    reusable_program_fingerprint = execution["reusable_program_fingerprint"]
    boundaries_raw = execution["declared_eager_boundary_node_ids"]
    require(
        isinstance(boundaries_raw, list),
        f"{label}.declared_eager_boundary_node_ids must be a list",
    )
    declared_eager_boundary_nodes = [
        text(
            node_id,
            f"{label}.declared_eager_boundary_node_ids[{index}]",
            portable=True,
        )
        for index, node_id in enumerate(boundaries_raw)
    ]
    require(
        declared_eager_boundary_nodes
        == sorted(set(declared_eager_boundary_nodes)),
        f"{label}.declared_eager_boundary_node_ids are not canonical",
    )
    declared_eager_boundary_node_set = set(declared_eager_boundary_nodes)
    require(
        declared_eager_boundary_node_set <= target_nodes,
        f"{label}.declared_eager_boundary_node_ids contain a non-target node",
    )
    if mode == "eager":
        require(
            compute_path_requirement == "eager_only"
            and reusable_program_fingerprint is None
            and not declared_eager_boundary_nodes,
            f"{label} eager execution has an invalid compute-path contract",
        )
    else:
        reusable_program_fingerprint = sha256_text(
            reusable_program_fingerprint,
            f"{label}.reusable_program_fingerprint",
        )
        if compute_path_requirement == "replayed_only":
            require(
                not declared_eager_boundary_nodes,
                f"{label} replayed-only execution declares eager boundaries",
            )
        elif compute_path_requirement == "replayed_with_declared_eager_boundaries":
            require(
                declared_eager_boundary_nodes
                and declared_eager_boundary_node_set < target_nodes,
                f"{label} mixed replay execution lacks both eager and replay nodes",
            )
        else:
            require(False, f"{label} replay execution has an invalid compute-path contract")
    sha256_text(execution["restore_sha256"], f"{label}.restore_sha256")
    initialization_identity = exact_object(
        execution["initialization_identity"],
        EXECUTION_INITIALIZATION_IDENTITY_FIELDS,
        f"{label}.initialization_identity",
    )
    for field in EXECUTION_INITIALIZATION_IDENTITY_FIELDS:
        sha256_text(
            initialization_identity[field],
            f"{label}.initialization_identity.{field}",
        )
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

    commands_raw = attribution["physical_commands"]
    require(
        isinstance(commands_raw, list) and commands_raw,
        f"{label}.attribution.physical_commands must be non-empty",
    )
    physical_commands: dict[int, dict[str, Any]] = {}
    compute_nodes: set[str] = set()
    actual_replayed_nodes: set[str] = set()
    observed_eager_boundary_nodes: set[str] = set()
    for index, raw in enumerate(commands_raw):
        command_label = f"{label}.attribution.physical_commands[{index}]"
        command = exact_object(raw, COMMAND_FIELDS, command_label)
        require(command["command_index"] == index, f"{command_label}.command_index is not contiguous")
        physical_commands[index] = command
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
                require(
                    node_id not in compute_nodes,
                    f"{command_label} duplicates eager compute attribution",
                )
                compute_nodes.add(node_id)
            elif node_id in declared_eager_boundary_node_set:
                require(
                    execution_path == "eager" and graph_nodes is None,
                    f"{command_label} declared eager boundary did not execute eagerly",
                )
                require(
                    node_id not in observed_eager_boundary_nodes,
                    f"{command_label} duplicates a declared eager boundary",
                )
                observed_eager_boundary_nodes.add(node_id)
                compute_nodes.add(node_id)
            else:
                require(
                    execution_path == "replayed",
                    f"{command_label} executed eagerly without a declared topology boundary",
                )

    replayed_segments_raw = attribution["replayed_segments"]
    require(
        isinstance(replayed_segments_raw, list),
        f"{label}.attribution.replayed_segments must be a list",
    )
    if mode == "eager":
        require(
            not replayed_segments_raw,
            f"{label} eager execution has replayed segment attribution",
        )
        require(
            not any(
                command["command_phase"] == "compute"
                and command["execution_path"] == "replayed"
                for command in physical_commands.values()
            ),
            f"{label} eager execution contains a replayed physical command",
        )
    else:
        require(
            replayed_segments_raw,
            f"{label} replay execution has no replayed segment attribution",
        )
        segment_physical_indices: set[int] = set()
        previous_physical_index: int | None = None
        for segment_index, raw in enumerate(replayed_segments_raw):
            segment_label = (
                f"{label}.attribution.replayed_segments[{segment_index}]"
            )
            segment = exact_object(raw, REPLAYED_SEGMENT_FIELDS, segment_label)
            physical_index = integer(
                segment["physical_command_index"],
                f"{segment_label}.physical_command_index",
            )
            require(
                previous_physical_index is None
                or previous_physical_index < physical_index,
                f"{label}.attribution.replayed_segments are not in physical command order",
            )
            previous_physical_index = physical_index
            require(
                physical_index not in segment_physical_indices,
                f"{segment_label} duplicates a physical command",
            )
            segment_physical_indices.add(physical_index)
            sha256_text(
                segment["reusable_program_fingerprint"],
                f"{segment_label}.reusable_program_fingerprint",
            )
            require(
                segment["reusable_program_fingerprint"]
                == reusable_program_fingerprint,
                f"{segment_label} references another reusable program",
            )
            sha256_text(
                segment["reusable_executable_fingerprint"],
                f"{segment_label}.reusable_executable_fingerprint",
            )
            physical = physical_commands.get(physical_index)
            require(
                physical is not None
                and physical["command_phase"] == "compute"
                and physical["execution_path"] == "replayed",
                f"{segment_label} is not bound to one physical replay command",
            )
            logical_raw = segment["logical_commands"]
            require(
                isinstance(logical_raw, list) and logical_raw,
                f"{segment_label}.logical_commands must be non-empty",
            )
            logical_graph_nodes = 0
            first_logical_node: str | None = None
            for logical_index, logical_value in enumerate(logical_raw):
                logical_label = f"{segment_label}.logical_commands[{logical_index}]"
                logical = exact_object(
                    logical_value,
                    REPLAYED_LOGICAL_COMMAND_FIELDS,
                    logical_label,
                )
                require(
                    logical["logical_command_ordinal"] == logical_index,
                    f"{logical_label}.logical_command_ordinal is not contiguous",
                )
                node_id = text(
                    logical["node_id"], f"{logical_label}.node_id", portable=True
                )
                require(
                    node_id in target_nodes,
                    f"{logical_label}.node_id is not a target node",
                )
                require(
                    node_id not in declared_eager_boundary_node_set,
                    f"{logical_label}.node_id is a declared eager boundary",
                )
                require(
                    node_id not in compute_nodes,
                    f"{logical_label}.node_id is attributed more than once",
                )
                actual_replayed_nodes.add(node_id)
                text(
                    logical["native_op_id"],
                    f"{logical_label}.native_op_id",
                    portable=True,
                )
                text(logical["batching_form"], f"{logical_label}.batching_form")
                require(
                    integer(
                        logical["participant_count"],
                        f"{logical_label}.participant_count",
                    )
                    == participant_count,
                    f"{logical_label} participant count differs from the case",
                )
                integer(logical["token_count"], f"{logical_label}.token_count")
                compute_dispatch_count = integer(
                    logical["compute_dispatch_count"],
                    f"{logical_label}.compute_dispatch_count",
                )
                transfer_command_count = integer(
                    logical["transfer_command_count"],
                    f"{logical_label}.transfer_command_count",
                )
                require(
                    compute_dispatch_count > 0 or transfer_command_count > 0,
                    f"{logical_label} contains no native work",
                )
                logical_graph_nodes += integer(
                    logical["reusable_graph_node_count"],
                    f"{logical_label}.reusable_graph_node_count",
                    minimum=1,
                )
                if first_logical_node is None:
                    first_logical_node = node_id
                compute_nodes.add(node_id)
            require(
                physical["node_id"] == first_logical_node,
                f"{segment_label} physical command is not bound to its first logical node",
            )
            require(
                physical["participant_count"] == participant_count
                and physical["reusable_graph_node_count"] == logical_graph_nodes,
                f"{segment_label} physical work differs from its logical attribution",
            )
        replayed_physical_indices = {
            index
            for index, command in physical_commands.items()
            if command["command_phase"] == "compute"
            and command["execution_path"] == "replayed"
        }
        require(
            replayed_physical_indices == segment_physical_indices,
            f"{label} replay physical commands and replayed segments differ",
        )
        require(
            observed_eager_boundary_nodes == declared_eager_boundary_node_set,
            f"{label} actual eager compute nodes differ from declared boundaries",
        )
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
    return execution, witnesses, actual_replayed_nodes


def validate_case(
    case: dict[str, Any],
    label: str,
    *,
    root_identity: dict[str, Any],
    denominator: dict[str, Any],
) -> dict[str, Any]:
    case = exact_object(case, CASE_FIELDS, label)
    require(case["schema_version"] == 2, f"{label}.schema_version must be 2")
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
    replay_shape_fingerprint: str | None = None
    restore_fingerprint: str | None = None
    actual_replayed_nodes: set[str] = set()
    for index, raw in enumerate(executions_raw):
        execution, witnesses, execution_replayed_nodes = validate_execution(
            raw,
            f"{label}.executions[{index}]",
            target_nodes=target_nodes,
            participant_count=participant_count,
            expected_witnesses=expected_witnesses,
            token_shape=case["token_shape"],
        )
        actual_replayed_nodes.update(execution_replayed_nodes)
        execution_id = execution["execution_id"]
        require(execution_id not in executions, f"{label} duplicates execution {execution_id}")
        topology = set(witnesses)
        if canonical_topology is None:
            canonical_topology = topology
        require(topology == canonical_topology, f"{label} execution witness topology drifted")
        if execution["mode"] == "replay":
            current = structural_sha256(
                {
                    "compute_path_requirement": execution["compute_path_requirement"],
                    "reusable_program_fingerprint": execution[
                        "reusable_program_fingerprint"
                    ],
                    "declared_eager_boundary_node_ids": execution[
                        "declared_eager_boundary_node_ids"
                    ],
                    "replayed_segments": execution["attribution"][
                        "replayed_segments"
                    ],
                }
            )
            if replay_shape_fingerprint is None:
                replay_shape_fingerprint = current
            require(
                current == replay_shape_fingerprint,
                f"{label} replay executable segment shape drifted",
            )
        require(
            execution["initialization_identity"]
            == {
                "input_sha256": initialization["input_sha256"],
                "rng_sha256": initialization["rng_sha256"],
                "initial_state_sha256": initialization["initial_state_sha256"],
            },
            f"{label} execution initialization identity differs from the exact restored bytes",
        )
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
    first_execution_id = min(witness_maps)
    return {
        "case_id": case_id,
        "model_key": model_key,
        "phase": phase,
        "partition": partition,
        "token_shape_fingerprint": structural_sha256(case["token_shape"]),
        "dtype": case["dtype"],
        "quantization": case["quantization"],
        "initialization_identity": (
            initialization["input_sha256"],
            initialization["rng_sha256"],
            initialization["initial_state_sha256"],
        ),
        "initial_state_kind": initial_state_kind,
        "workspace_poison": poison,
        "target_node_coverage": target_node_coverage,
        "target_signature": tuple(
            sorted(
                (requirement_key, tuple(sorted(nodes)))
                for requirement_key, nodes in target_node_coverage.items()
            )
        ),
        "target_keys": set(target_node_coverage),
        "target_nodes": target_nodes,
        "actual_replayed_nodes": actual_replayed_nodes,
        "canonical_witnesses": witness_maps[first_execution_id],
        "state_witness": state_witness,
        "execution_count": len(executions),
        "comparison_count": len(comparisons_raw),
        "witness_count": len(canonical_topology or set()) * len(executions),
    }


def validate_artifact(
    root: Path,
    expected_source: dict[str, Any],
    expected_scope: str = RELEASE_SCOPE,
    *,
    allow_internal_fixture: bool = False,
) -> dict[str, Any]:
    contract = scope_contract(expected_scope)
    expected_models = set(contract["models"])
    required_partitions = set(contract["partitions"])
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
    require(
        text(manifest["scope"], "evidence.scope") == expected_scope,
        "evidence.scope differs from the requested validator scope",
    )

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
    require(
        binary_ref["path"] == "binary/ferrum",
        "evidence.source.binary must use the canonical binary/ferrum path",
    )
    binary_sha256 = binary_ref["sha256"]

    device_fingerprint, hardware_id = validate_hardware(
        root, manifest["hardware"], source=source
    )
    candidate_build_receipt = validate_candidate_build_provenance(
        root,
        source,
        hardware_id=hardware_id,
        binary_ref=binary_ref,
        allow_internal_fixture=allow_internal_fixture,
    )
    locked_models = validate_models_lock(root, manifest["models_lock"])
    verified_model_dirs = validate_model_verification(
        root,
        manifest["model_verification"],
        expected_scope=expected_scope,
        expected_models=expected_models,
        expected_source=source,
        locked_models=locked_models,
    )

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
        read_json(denominator_path, max_bytes=MAX_DENOMINATOR_JSON_BYTES),
        expected_models,
        expected_scope,
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
    require(
        model_keys == sorted(expected_models),
        "evidence.models differ from the selected scope model set",
    )

    runner = exact_object(manifest["runner"], RUNNER_FIELDS, "evidence.runner")
    validate_runner_environment(runner["environment"])
    repository_root = Path(
        text(runner["repository_root"], "evidence.runner.repository_root")
    )
    require(
        repository_root.is_absolute(),
        "evidence.runner.repository_root must be absolute",
    )
    candidate_repository_root = Path(
        text(
            candidate_build_receipt["repository_root"],
            "candidate build repository_root",
        )
    )
    require(
        repository_root == candidate_repository_root,
        "runner repository root differs from the candidate build repository root",
    )
    require(isinstance(runner["command"], list) and runner["command"], "evidence.runner.command must be non-empty")
    runner_command = [
        text(item, f"evidence.runner.command[{index}]")
        for index, item in enumerate(runner["command"])
    ]
    require(
        runner_command[0] == source["binary_path"],
        "evidence.runner.command must execute the recorded release binary",
    )
    canonical_prefix = [
        source["binary_path"],
        "vnext-determinism",
        "--models-lock",
        runner_command[3] if len(runner_command) > 3 else "",
        "--artifact-root",
        runner_command[5] if len(runner_command) > 5 else "",
    ]
    require(
        runner_command[:6] == canonical_prefix,
        "evidence.runner.command is not the canonical vnext-determinism collector prefix",
    )
    recorded_artifact_root = Path(runner_command[5])
    require(
        recorded_artifact_root.is_absolute()
        and Path(runner_command[3]) == recorded_artifact_root / "models.lock.json"
        and Path(source["binary_path"])
        == recorded_artifact_root / "binary" / "ferrum",
        "runner binary, model lock and artifact root do not share the canonical layout",
    )
    argument_offset = 6
    if expected_scope == M1_S2_FOCUSED_SCOPE:
        require(
            runner_command[6:8] == ["--scope", M1_S2_FOCUSED_SCOPE],
            "focused evidence.runner.command lacks the explicit typed scope",
        )
        argument_offset = 8
    else:
        require(
            "--scope" not in runner_command,
            "release-full evidence must use the default canonical collector scope",
        )
    model_tail = runner_command[argument_offset:]
    require(
        len(model_tail) == 2 * len(expected_models)
        and model_tail[::2] == ["--model"] * len(expected_models),
        "evidence.runner.command has a non-canonical model binding tail",
    )
    model_arguments = model_tail[1::2]
    require(
        sorted(argument.split("=", 1)[0] for argument in model_arguments)
        == sorted(expected_models)
        and all("=" in argument and argument.split("=", 1)[1] for argument in model_arguments),
        "evidence.runner.command model bindings differ from the selected scope",
    )
    runner_model_dirs = {
        argument.split("=", 1)[0]: argument.split("=", 1)[1]
        for argument in model_arguments
    }
    require(
        runner_model_dirs == verified_model_dirs,
        "runner model directories differ from the hash-verified directories",
    )
    started = validate_timestamp(runner["started_at"], "evidence.runner.started_at")
    finished = validate_timestamp(runner["finished_at"], "evidence.runner.finished_at")
    require(
        datetime.fromisoformat(finished.replace("Z", "+00:00"))
        > datetime.fromisoformat(started.replace("Z", "+00:00")),
        "evidence.runner timestamps are not increasing",
    )
    require(runner["exit_code"] == 0, "evidence.runner.exit_code must be 0")
    receipt_ref, receipt_path = validate_file_ref(
        root,
        runner["receipt"],
        "evidence.runner.receipt",
        max_size_bytes=MAX_RECEIPT_JSON_BYTES,
    )
    stdout_ref, stdout_path = validate_file_ref(
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
        allow_empty=True,
    )
    require(
        receipt_ref["path"] == "runner/receipt.json"
        and stdout_ref["path"] == "runner/stdout.log"
        and stderr_ref["path"] == "runner/stderr.log",
        "runner evidence references do not use the canonical runner paths",
    )
    validate_bounded_receipt(
        read_json(receipt_path, max_bytes=MAX_RECEIPT_JSON_BYTES),
        command=runner_command,
        repository_root=repository_root,
        recorded_artifact_root=recorded_artifact_root,
        started_at=started,
        finished_at=finished,
        exit_code=runner["exit_code"],
        stdout_ref=stdout_ref,
        stderr_ref=stderr_ref,
    )

    case_refs = manifest["cases"]
    expected_case_count = len(expected_models) * len(required_partitions) * 4
    require(
        isinstance(case_refs, list)
        and len(case_refs) == expected_case_count
        and len(case_refs) <= MAX_CASES,
        "evidence.cases cardinality is invalid",
    )
    collector, collector_models = validate_collector_manifest(
        root,
        manifest["collector"],
        scope=expected_scope,
        expected_models=expected_models,
        runner_command=runner_command,
        runner_stdout_path=stdout_path,
        binary_ref=binary_ref,
        models_lock_ref=manifest["models_lock"],
        hardware_probe_ref=manifest["hardware"]["probe"],
        device_fingerprint=device_fingerprint,
        denominator_ref=manifest["denominator"],
        denominator=denominator,
        model_directories=verified_model_dirs,
        case_refs=case_refs,
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
        require(
            ref["path"] == f"cases/{case['case_id']}.json",
            "case reference path differs from its canonical case_id path",
        )
        collector_model = collector_models[case["model_key"]]
        require(
            case["dtype"] == collector_model["dtype"]
            and case["quantization"] == collector_model["quantization"],
            "case dtype or quantization differs from collector model identity",
        )
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
        key: set() for key in expected_models
    }
    state_by_model: dict[str, set[str]] = {key: set() for key in expected_models}
    coverage_nodes: dict[tuple[str, tuple[str, str]], set[str]] = {}
    actual_replayed_nodes: dict[tuple[str, tuple[str, str]], set[str]] = {}
    poisons: dict[tuple[str, tuple[str, str]], set[str]] = {}
    phases: dict[tuple[str, tuple[str, str]], set[str]] = {}
    poison_pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for case in cases:
        model_key = case["model_key"]
        partitions_by_model[model_key].add((case["phase"], case["partition"]))
        if case["state_witness"] and case["initial_state_kind"] in {"zero", "nonzero"}:
            state_by_model[model_key].add(case["initial_state_kind"])
        for requirement_key, nodes in case["target_node_coverage"].items():
            selection_key = (model_key, requirement_key)
            coverage_nodes.setdefault(selection_key, set()).update(nodes)
            actual_replayed_nodes.setdefault(selection_key, set()).update(
                nodes & case["actual_replayed_nodes"]
            )
            poisons.setdefault(selection_key, set()).add(case["workspace_poison"])
            phases.setdefault(selection_key, set()).add(case["phase"])
        poison_pair_key = (
            model_key,
            case["phase"],
            case["partition"],
            case["token_shape_fingerprint"],
            case["dtype"],
            case["quantization"],
            case["initial_state_kind"],
            case["target_signature"],
        )
        poison_cases = poison_pairs.setdefault(poison_pair_key, {})
        require(
            case["workspace_poison"] not in poison_cases,
            f"duplicate workspace poison fixture for {poison_pair_key}",
        )
        poison_cases[case["workspace_poison"]] = case

    for poison_pair_key, poison_cases in poison_pairs.items():
        require(
            set(poison_cases) == {"00", "a5"},
            f"workspace poison fixture lacks an exact counterpart for {poison_pair_key}",
        )
        zero_poison = poison_cases["00"]
        a5_poison = poison_cases["a5"]
        require(
            zero_poison["initialization_identity"] == a5_poison["initialization_identity"],
            f"workspace poison fixtures changed input/RNG/initial-state identity for {poison_pair_key}",
        )
        require(
            zero_poison["canonical_witnesses"] == a5_poison["canonical_witnesses"],
            f"workspace poison changed raw witness bytes for {poison_pair_key}",
        )

    for model_key in expected_models:
        require(
            partitions_by_model[model_key] == required_partitions,
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
            if requirement["replay_equivalence"] == "bitwise_eager_equivalent":
                require(
                    actual_replayed_nodes.get(selection_key)
                    == set(selection["node_ids"]),
                    f"provider selection {selection_key} lacks actual replay coverage",
                )
            require(
                poisons.get(selection_key) == {"00", "a5"},
                f"provider selection {selection_key} lacks both workspace poison patterns",
            )
            require(
                phases.get(selection_key) == {"prefill", "decode"},
                f"provider selection {selection_key} lacks prefill/decode proof",
            )
            for phase, partition in required_partitions:
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

    execution_count = sum(case["execution_count"] for case in cases)
    comparison_count = sum(case["comparison_count"] for case in cases)
    require(
        collector["execution_count"] == execution_count
        and collector["comparison_count"] == comparison_count,
        "collector execution/comparison counts differ from validated cases",
    )

    return {
        "source_git_sha": git_sha,
        "source_tree_sha": git_tree_sha,
        "binary_sha256": binary_sha256,
        "denominator_fingerprint": denominator_fingerprint,
        "device_fingerprint": device_fingerprint,
        "scope": expected_scope,
        "model_keys": sorted(expected_models),
        "provider_requirement_count": len(denominator["requirements"]),
        "case_count": len(cases),
        "execution_count": execution_count,
        "comparison_count": comparison_count,
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
    status = git_stdout("status", "--short", "--untracked-files=all")
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
    scope = text(summary.get("scope"), "validation summary scope")
    contract = scope_contract(scope)
    pass_line = f"{contract['pass_prefix']}: {out_dir}"
    return {
        "schema_version": 1,
        "artifact_type": VALIDATOR_ARTIFACT_TYPE,
        "lane": contract["lane"],
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
    scope: str,
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
                "lane": scope_contract(scope)["lane"],
                "scope": scope,
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
    model_keys: set[str],
    scope: str,
) -> tuple[dict[str, Any], str, dict[tuple[str, str], str]]:
    models = []
    primary_selections = []
    secondary_selections = []
    for index, model_key in enumerate(sorted(model_keys), 1):
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
        "schema_version": {"major": 1, "minor": 1},
        "provider_coverage": (
            "selected_plan_providers"
            if scope == M1_S2_FOCUSED_SCOPE
            else "all_catalog_providers"
        ),
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
    mixed_replay = (
        replay and model_key == "m1-qwen35-4b" and partition == "c1"
    )
    declared_eager_boundary_node_ids = [node_ids[-1]] if mixed_replay else []
    reusable_program_fingerprint = (
        hashlib.sha256(f"{model_key}/{partition}/program".encode()).hexdigest()
        if replay
        else None
    )
    physical_commands = [
        {
            "command_index": command_index,
            "node_id": node_id,
            "command_phase": "compute",
            "native_op_id": "native.test",
            "execution_path": (
                "eager"
                if not replay or node_id in declared_eager_boundary_node_ids
                else "replayed"
            ),
            "batching_form": "scalar",
            "participant_count": participant_count,
            "token_count": participant_count,
            "compute_dispatch_count": 1,
            "transfer_command_count": 0,
            "reusable_graph_node_count": (
                1
                if replay and node_id not in declared_eager_boundary_node_ids
                else None
            ),
        }
        for command_index, node_id in enumerate(node_ids)
    ]
    replayed_segments = (
        [
            {
                "physical_command_index": command_index,
                "reusable_program_fingerprint": reusable_program_fingerprint,
                "reusable_executable_fingerprint": hashlib.sha256(
                    f"{model_key}/{partition}/segment/{command_index}".encode()
                ).hexdigest(),
                "logical_commands": [
                    {
                        "logical_command_ordinal": 0,
                        "node_id": node_id,
                        "native_op_id": "native.test",
                        "batching_form": "scalar",
                        "participant_count": participant_count,
                        "token_count": participant_count,
                        "compute_dispatch_count": 1,
                        "transfer_command_count": 0,
                        "reusable_graph_node_count": 1,
                    }
                ],
            }
            for command_index, node_id in enumerate(node_ids)
            if node_id not in declared_eager_boundary_node_ids
        ]
        if replay
        else []
    )
    return {
        "execution_id": execution_id,
        "mode": mode,
        "compute_path_requirement": (
            "replayed_with_declared_eager_boundaries"
            if mixed_replay
            else "replayed_only"
            if replay
            else "eager_only"
        ),
        "reusable_program_fingerprint": reusable_program_fingerprint,
        "declared_eager_boundary_node_ids": declared_eager_boundary_node_ids,
        "restore_sha256": "1" * 64,
        "initialization_identity": {
            "input_sha256": "4" * 64,
            "rng_sha256": "5" * 64,
            "initial_state_sha256": "6" * 64,
        },
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
            "physical_commands": physical_commands,
            "replayed_segments": replayed_segments,
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
        immediate = [4]
        starts = [4]
    elif partition == "multi_participant":
        participants = 4
        immediate = [1] * participants
        starts = [8] * participants
    else:
        participants = 32
        immediate = [1] * participants
        starts = [8] * participants
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
        "schema_version": 2,
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
            {
                "path": "tokenizer.json",
                "sha256": f"{index + 6}" * 64,
                "size_bytes": 2048 + index,
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
    scope: str = RELEASE_SCOPE,
) -> None:
    contract = scope_contract(scope)
    selected_models = set(contract["models"])
    selected_partitions = set(contract["partitions"])
    denominator, denominator_fingerprint, witness_fingerprints = (
        make_selftest_denominator(root, selected_models, scope)
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
    runner_root = root / "runner"
    runner_root.mkdir()
    stdout_path = runner_root / "stdout.log"
    stderr_path = runner_root / "stderr.log"
    stderr_path.write_bytes(b"")
    case_refs = []
    case_shapes = sorted(selected_partitions)
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
        "/workspace/artifacts/binary/ferrum",
        "vnext-determinism",
        "--models-lock",
        "/workspace/artifacts/models.lock.json",
        "--artifact-root",
        "/workspace/artifacts",
    ]
    if scope == M1_S2_FOCUSED_SCOPE:
        runner_command.extend(["--scope", scope])
    for model_key in sorted(selected_models):
        runner_command.extend(
            ["--model", f"{model_key}=/workspace/models/{model_key}"]
        )
    collector_pass = exact_collector_pass_line(scope, runner_command[5])
    stdout_path.write_text(collector_pass + "\n", encoding="utf-8")
    model_verification_path = root / "model-verification.json"
    write_json(
        model_verification_path,
        {
            "schema_version": 1,
            "artifact_type": "runtime_vnext_cuda_determinism_model_verification",
            "scope": scope,
            "source_git_sha": expected_source["git_sha"],
            "source_tree_sha": expected_source["git_tree_sha"],
            "collector": {
                "path": EVIDENCE_COLLECTOR_PATH.relative_to(REPO_ROOT).as_posix(),
                "sha256": file_sha256(EVIDENCE_COLLECTOR_PATH),
            },
            "verified_at": "2026-07-27T00:00:00Z",
            "models": [
                {
                    "model_key": model_key,
                    "model_dir": f"/workspace/models/{model_key}",
                    "files": locked_models_by_key[model_key]["files"],
                    "consumed_files": [
                        "config.json",
                        "model.safetensors",
                        "tokenizer.json",
                    ],
                }
                for model_key in sorted(selected_models)
            ],
        },
    )
    started_at = "2026-07-27T00:00:00Z"
    finished_at = "2026-07-27T00:01:00Z"
    repository_root = REPO_ROOT.resolve()
    runner_environment = canonical_determinism_environment(
        {"HOME": "/workspace", "PATH": "/usr/local/bin:/usr/bin:/bin"}
    )
    receipt = {
        "schema": "ferrum.bounded-command-receipt.v1",
        "command": runner_command,
        "cwd": str(repository_root),
        "pid": 4242,
        "pgid": 4242,
        "limits": {
            "wall_timeout_seconds": 3600.0,
            "max_processes": 16,
            "max_group_threads": 128,
            "max_per_process_threads": 64,
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
            "path": "/workspace/artifacts/runner/stdout.log",
            "sha256": file_sha256(stdout_path),
            "size_bytes": stdout_path.stat().st_size,
        },
        "stderr": {
            "path": "/workspace/artifacts/runner/stderr.log",
            "sha256": file_sha256(stderr_path),
            "size_bytes": stderr_path.stat().st_size,
        },
    }
    receipt_path = runner_root / "receipt.json"
    write_json(receipt_path, receipt)
    build_provenance_root = root / BUILD_PROVENANCE_ROOT
    candidate_binary = build_provenance_root / "build/candidate/ferrum"
    candidate_binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary_path, candidate_binary)
    baseline_scenarios.make_candidate_build_receipt_fixture(
        build_provenance_root,
        {
            "backend": "cuda",
            "source_git_sha": expected_source["git_sha"],
            "source_tree_sha": expected_source["git_tree_sha"],
            "hardware_id": "selftest-cuda-rtx4090",
            "binary_artifact": baseline_scenarios.existing_artifact_ref(
                build_provenance_root,
                candidate_binary,
                "binary",
            ),
            "binary_sha256": binary_ref["sha256"],
        },
    )
    fixture_lock = root / NATIVE_OPERATOR_SET_LOCK_PATH
    require(fixture_lock.is_file(), "candidate fixture native operator lock is missing")

    evidence_models_by_key = {row["model_key"]: row for row in models}
    collector_models = []
    for model_key in sorted(selected_models):
        sample_case_ref = next(
            ref
            for ref in case_refs
            if ref["path"].startswith(f"cases/{model_key}.")
        )
        sample_case = read_json(root / sample_case_ref["path"])
        planned = evidence_models_by_key[model_key]
        collector_models.append(
            {
                "model_key": model_key,
                "model_dir": f"/workspace/models/{model_key}",
                "resolved_plan_fingerprint": planned[
                    "resolved_plan_fingerprint"
                ],
                "plan_hash": planned["plan_hash"],
                "dtype": sample_case["dtype"],
                "quantization": sample_case["quantization"],
                "case_count": len(selected_partitions) * 4,
            }
        )
    collector = {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_cuda_determinism_collector",
        "status": "pass",
        "backend": "cuda",
        "scope": scope,
        "models_lock": models_lock_ref,
        "hardware_probe": hardware_ref,
        "device_fingerprint": device_fingerprint,
        "binary": {
            "path": runner_command[0],
            "sha256": binary_ref["sha256"],
            "size_bytes": binary_ref["size_bytes"],
        },
        "denominator": {
            **denominator_ref,
            "fingerprint": denominator_fingerprint,
        },
        "models": collector_models,
        "cases": case_refs,
        "case_count": len(case_refs),
        "execution_count": len(case_refs) * 12,
        "comparison_count": len(case_refs) * 15,
        "pass_line": collector_pass,
    }
    collector_path = root / "collector.json"
    write_json(collector_path, collector)
    evidence = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "backend": "cuda",
        "scope": scope,
        "source": {
            "git_sha": expected_source["git_sha"],
            "git_tree_sha": expected_source["git_tree_sha"],
            "dirty_status": [],
            "binary_path": "/workspace/artifacts/binary/ferrum",
            "binary": binary_ref,
            "candidate_build_receipt": selftest_file_ref(
                root,
                root / CANDIDATE_BUILD_RECEIPT_PATH,
            ),
            "native_operator_set_lock": selftest_file_ref(
                root,
                root / NATIVE_OPERATOR_SET_LOCK_PATH,
            ),
        },
        "hardware": {
            "probe": hardware_ref,
            "fingerprint": device_fingerprint,
        },
        "models_lock": models_lock_ref,
        "model_verification": selftest_file_ref(root, model_verification_path),
        "collector": selftest_file_ref(root, collector_path),
        "denominator": {
            **denominator_ref,
            "fingerprint": denominator_fingerprint,
        },
        "models": models,
        "runner": {
            "command": runner_command,
            "environment": runner_environment,
            "repository_root": str(repository_root),
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
    collector_path = root / "collector.json"
    collector = read_json(collector_path)
    relative = case_path.relative_to(root).as_posix()
    refreshed = selftest_file_ref(root, case_path)
    for refs in (evidence["cases"], collector["cases"]):
        for index, ref in enumerate(refs):
            if ref["path"] == relative:
                refs[index] = refreshed
                break
    write_json(collector_path, collector)
    evidence["collector"] = selftest_file_ref(root, collector_path)
    write_json(root / "evidence.json", evidence)


def run_self_test() -> None:
    expected_source = {
        "git_sha": git_stdout("rev-parse", "HEAD"),
        "git_tree_sha": git_stdout("rev-parse", "HEAD^{tree}"),
    }
    with tempfile.TemporaryDirectory(prefix="ferrum-vnext-cuda-determinism-") as temporary:
        base = Path(temporary) / "base"
        base.mkdir()
        make_selftest_artifact(base, expected_source)
        summary = validate_artifact(
            base,
            expected_source,
            allow_internal_fixture=True,
        )
        require(summary["case_count"] == 72, "self-test case denominator drifted")
        require(summary["comparison_count"] == 1080, "self-test comparison denominator drifted")
        plan_ordered_denominator = read_json(base / "denominator.json")
        plan_ordered_denominator["coverage"]["models"][0]["node_ids"].reverse()
        validate_denominator(
            plan_ordered_denominator,
            PRIMARY_MODELS,
            RELEASE_SCOPE,
        )
        plan_ordered_nodes = ["node.layer.2", "node.layer.10"]
        require(
            validate_string_list(
                plan_ordered_nodes,
                "self-test plan-ordered nodes",
                portable=True,
                canonical_order=False,
            )
            == plan_ordered_nodes,
            "plan-ordered node validation reordered the execution topology",
        )
        immediate_span_location = selftest_location(
            M1_MODEL,
            "primary",
            kind="output",
            index=0,
        )
        immediate_span_location["declared_length_bytes"] = 64
        immediate_span_location["extent"] = {
            "kind": "immediate_token_span",
            "bytes_per_token": 4,
            "maximum_tokens": 4,
        }
        validate_location(
            immediate_span_location,
            "self-test immediate token span",
        )

        focused = Path(temporary) / "focused"
        focused.mkdir()
        make_selftest_artifact(focused, expected_source, M1_S2_FOCUSED_SCOPE)
        focused_summary = validate_artifact(
            focused,
            expected_source,
            M1_S2_FOCUSED_SCOPE,
            allow_internal_fixture=True,
        )
        require(
            focused_summary["model_keys"] == [M1_MODEL]
            and focused_summary["case_count"] == 20
            and focused_summary["comparison_count"] == 300,
            "focused self-test denominator drifted",
        )
        focused_denominator = read_json(focused / "denominator.json")
        validate_denominator(
            focused_denominator,
            {M1_MODEL},
            M1_S2_FOCUSED_SCOPE,
        )
        wrong_provider_coverage = copy.deepcopy(focused_denominator)
        wrong_provider_coverage["provider_coverage"] = "all_catalog_providers"
        try:
            validate_denominator(
                wrong_provider_coverage,
                {M1_MODEL},
                M1_S2_FOCUSED_SCOPE,
            )
        except DeterminismArtifactError as error:
            require(
                "provider_coverage differs" in str(error),
                f"focused provider coverage was rejected for the wrong reason: {error}",
            )
        else:
            raise AssertionError(
                "focused denominator accepted release-full provider coverage"
            )
        try:
            validate_artifact(
                focused,
                expected_source,
                RELEASE_SCOPE,
                allow_internal_fixture=True,
            )
        except DeterminismArtifactError as error:
            require(
                "scope differs" in str(error),
                f"focused artifact was rejected for the wrong full-gate reason: {error}",
            )
        else:
            raise AssertionError("focused artifact unexpectedly passed the release-full gate")

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
        first_a5_case = "cases/m1-qwen35-4b.prefill.single_token.zero.a5.json"
        replay_case = "cases/m1-qwen35-4b.decode.c1.zero.00.json"

        def mutate_receipt_sampling_error(root: Path) -> None:
            receipt_path = root / "runner/receipt.json"
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

        def mutate_candidate_build_receipt(root: Path) -> None:
            receipt_path = root / CANDIDATE_BUILD_RECEIPT_PATH
            receipt = read_json(receipt_path)
            receipt["returncode"] = 1
            write_json(receipt_path, receipt)
            evidence = read_json(root / "evidence.json")
            evidence["source"]["candidate_build_receipt"] = selftest_file_ref(
                root, receipt_path
            )
            write_json(root / "evidence.json", evidence)

        def mutate_collector_case_count(root: Path) -> None:
            collector_path = root / "collector.json"
            collector = read_json(collector_path)
            collector["case_count"] += 1
            write_json(collector_path, collector)
            evidence = read_json(root / "evidence.json")
            evidence["collector"] = selftest_file_ref(root, collector_path)
            write_json(root / "evidence.json", evidence)

        def mutate_runner_environment(root: Path) -> None:
            evidence = read_json(root / "evidence.json")
            evidence["runner"]["environment"]["RAYON_NUM_THREADS"] = "64"
            write_json(root / "evidence.json", evidence)

        def mutate_runner_undeclared_environment(root: Path) -> None:
            evidence = read_json(root / "evidence.json")
            evidence["runner"]["environment"]["LD_PRELOAD"] = "/tmp/forged.so"
            write_json(root / "evidence.json", evidence)

        def mutate_runner_repository_root(root: Path) -> None:
            evidence = read_json(root / "evidence.json")
            evidence["runner"]["repository_root"] = "/workspace/forged"
            write_json(root / "evidence.json", evidence)

        def mutate_runner_repository_root_and_receipt(root: Path) -> None:
            mutate_runner_repository_root(root)
            receipt_path = root / "runner/receipt.json"
            receipt = read_json(receipt_path)
            receipt["cwd"] = "/workspace/forged"
            write_json(receipt_path, receipt)
            evidence = read_json(root / "evidence.json")
            evidence["runner"]["receipt"] = selftest_file_ref(root, receipt_path)
            write_json(root / "evidence.json", evidence)

        def mutate_runner_stdout_reference(root: Path) -> None:
            redirected = root / "redirected/stdout.log"
            redirected.parent.mkdir()
            shutil.copy2(root / "runner/stdout.log", redirected)
            evidence = read_json(root / "evidence.json")
            evidence["runner"]["stdout"] = selftest_file_ref(root, redirected)
            write_json(root / "evidence.json", evidence)

        def mutate_receipt_stdout_path(root: Path) -> None:
            receipt_path = root / "runner/receipt.json"
            receipt = read_json(receipt_path)
            receipt["stdout"]["path"] = "/workspace/forged/stdout.log"
            write_json(receipt_path, receipt)
            evidence = read_json(root / "evidence.json")
            evidence["runner"]["receipt"] = selftest_file_ref(root, receipt_path)
            write_json(root / "evidence.json", evidence)

        def mutate_models_lock_catalog(root: Path) -> None:
            lock_path = root / "models.lock.json"
            lock = read_json(lock_path)
            lock["catalog_sha256"] = "0" * 64
            write_json(lock_path, lock)
            evidence = read_json(root / "evidence.json")
            evidence["models_lock"] = selftest_file_ref(root, lock_path)
            write_json(root / "evidence.json", evidence)

        def mutate_verified_model_directory(root: Path) -> None:
            verification_path = root / "model-verification.json"
            verification = read_json(verification_path)
            verification["models"][0]["model_dir"] = "/workspace/models/forged"
            write_json(verification_path, verification)
            evidence = read_json(root / "evidence.json")
            evidence["model_verification"] = selftest_file_ref(
                root, verification_path
            )
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

        def mutate_replay_fallback(value: dict[str, Any]) -> None:
            execution = value["executions"][-1]
            for command in execution["attribution"]["physical_commands"]:
                if (
                    command["command_phase"] == "compute"
                    and command["execution_path"] == "replayed"
                ):
                    command.update(
                        {
                            "execution_path": "eager",
                            "reusable_graph_node_count": None,
                        }
                    )
                    return
            raise AssertionError("self-test fixture has no physical replay command")

        def mutate_boundary_replayed(value: dict[str, Any]) -> None:
            execution = value["executions"][-1]
            boundary = execution["declared_eager_boundary_node_ids"][0]
            for command in execution["attribution"]["physical_commands"]:
                if command["node_id"] == boundary:
                    command.update(
                        {
                            "execution_path": "replayed",
                            "reusable_graph_node_count": 1,
                        }
                    )
                    return
            raise AssertionError("self-test fixture has no declared eager boundary")

        def mutate_segment_program_fingerprint(value: dict[str, Any]) -> None:
            execution = value["executions"][-1]
            execution["attribution"]["replayed_segments"][0][
                "reusable_program_fingerprint"
            ] = "0" * 64

        mutations.extend(
            [
                (
                    "binary-tamper",
                    mutate_binary_bytes,
                    "source.binary.size_bytes differs",
                ),
                (
                    "candidate-build-receipt-tamper",
                    mutate_candidate_build_receipt,
                    "candidate build returncode must be 0",
                ),
                (
                    "collector-case-count-tamper",
                    mutate_collector_case_count,
                    "collector case references or exact denominator differ",
                ),
                (
                    "runner-worker-environment-tamper",
                    mutate_runner_environment,
                    "fixed CUDA/worker limits",
                ),
                (
                    "runner-undeclared-environment-tamper",
                    mutate_runner_undeclared_environment,
                    "undeclared variable",
                ),
                (
                    "runner-repository-root-tamper",
                    mutate_runner_repository_root,
                    "candidate build repository root",
                ),
                (
                    "runner-repository-root-joint-tamper",
                    mutate_runner_repository_root_and_receipt,
                    "candidate build repository root",
                ),
                (
                    "runner-stdout-reference-redirect",
                    mutate_runner_stdout_reference,
                    "canonical runner paths",
                ),
                (
                    "runner-receipt-stdout-path-tamper",
                    mutate_receipt_stdout_path,
                    "canonical artifact path",
                ),
                (
                    "models-lock-stale",
                    mutate_models_lock_catalog,
                    "models.lock is stale",
                ),
                (
                    "verified-model-directory-drift",
                    mutate_verified_model_directory,
                    "runner model directories differ",
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
                    "shape-fixture-drift",
                    mutate_case(
                        first_case,
                        lambda value: value["token_shape"].update(
                            {
                                "immediate_tokens": [2],
                                "source_end_tokens": [2],
                            }
                        ),
                    ),
                    "canonical Rust shape fixture",
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
                        mutate_replay_fallback,
                    ),
                    "executed eagerly without a declared topology boundary",
                ),
                (
                    "missing-eager-boundary-declaration",
                    mutate_case(
                        replay_case,
                        lambda value: value["executions"][-1].update(
                            {"declared_eager_boundary_node_ids": []}
                        ),
                    ),
                    "lacks both eager and replay nodes",
                ),
                (
                    "declared-boundary-replayed",
                    mutate_case(replay_case, mutate_boundary_replayed),
                    "declared eager boundary did not execute eagerly",
                ),
                (
                    "segment-program-drift",
                    mutate_case(replay_case, mutate_segment_program_fingerprint),
                    "references another reusable program",
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
                    "cross-poison-raw-mismatch",
                    mutate_case(
                        first_a5_case,
                        lambda value: [
                            execution["witnesses"][0].update({"raw_sha256": "0" * 64})
                            for execution in value["executions"]
                        ],
                    ),
                    "workspace poison changed raw witness bytes",
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
                    "invented-initialization-identity",
                    mutate_case(
                        first_case,
                        lambda value: value["executions"][0][
                            "initialization_identity"
                        ].update({"input_sha256": "0" * 64}),
                    ),
                    "differs from the exact restored bytes",
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
                paths = [root / ref["path"] for ref in evidence["cases"]]
                for path in paths:
                    value = read_json(path)
                    mutation(value)
                    write_json(path, value)
                    refresh_case_ref(root, path)

            return apply

        def drop_secondary_provider(value: dict[str, Any]) -> None:
            secondary_nodes = set(value["coverage_targets"][1]["node_ids"])
            value["coverage_targets"] = value["coverage_targets"][:1]
            for execution in value["executions"]:
                execution["declared_eager_boundary_node_ids"] = [
                    node_id
                    for node_id in execution["declared_eager_boundary_node_ids"]
                    if node_id not in secondary_nodes
                ]
                if (
                    execution["mode"] == "replay"
                    and not execution["declared_eager_boundary_node_ids"]
                ):
                    execution["compute_path_requirement"] = "replayed_only"
                attribution = execution["attribution"]
                commands = []
                command_index_map = {}
                for command in attribution["physical_commands"]:
                    if command["node_id"] in secondary_nodes:
                        continue
                    old_index = command["command_index"]
                    new_index = len(commands)
                    command_index_map[old_index] = new_index
                    command["command_index"] = new_index
                    commands.append(command)
                attribution["physical_commands"] = commands
                segments = []
                for segment in attribution["replayed_segments"]:
                    old_index = segment["physical_command_index"]
                    if old_index not in command_index_map:
                        continue
                    logical_commands = [
                        command
                        for command in segment["logical_commands"]
                        if command["node_id"] not in secondary_nodes
                    ]
                    if not logical_commands:
                        continue
                    for ordinal, command in enumerate(logical_commands):
                        command["logical_command_ordinal"] = ordinal
                    segment["physical_command_index"] = command_index_map[old_index]
                    segment["logical_commands"] = logical_commands
                    segments.append(segment)
                attribution["replayed_segments"] = segments
                execution["witnesses"] = [
                    witness
                    for witness in execution["witnesses"]
                    if witness["node_id"] not in secondary_nodes
                ]

        def make_m1_secondary_always_eager(value: dict[str, Any]) -> None:
            if value["model_key"] != M1_MODEL:
                return
            boundary = f"node.{M1_MODEL}.secondary"
            for execution in value["executions"]:
                if execution["mode"] != "replay":
                    continue
                execution["compute_path_requirement"] = (
                    "replayed_with_declared_eager_boundaries"
                )
                execution["declared_eager_boundary_node_ids"] = [boundary]
                for command in execution["attribution"]["physical_commands"]:
                    if command["node_id"] == boundary:
                        command["execution_path"] = "eager"
                        command["reusable_graph_node_count"] = None
                retained_segments = []
                for segment in execution["attribution"]["replayed_segments"]:
                    segment["logical_commands"] = [
                        command
                        for command in segment["logical_commands"]
                        if command["node_id"] != boundary
                    ]
                    if not segment["logical_commands"]:
                        continue
                    for ordinal, command in enumerate(segment["logical_commands"]):
                        command["logical_command_ordinal"] = ordinal
                    retained_segments.append(segment)
                execution["attribution"]["replayed_segments"] = retained_segments

        mutations.extend(
            [
                (
                    "missing-c32",
                    remove_cases(lambda path: ".decode.c32." in path),
                    "cardinality is invalid",
                ),
                (
                    "missing-provider",
                    mutate_all_cases(drop_secondary_provider),
                    "lacks exact node proof coverage",
                ),
                (
                    "single-poison",
                    remove_cases(lambda path: path.endswith(".a5.json")),
                    "cardinality is invalid",
                ),
                (
                    "zero-state-only",
                    remove_cases(lambda path: ".nonzero." in path),
                    "cardinality is invalid",
                ),
                (
                    "missing-actual-replay-coverage",
                    mutate_all_cases(make_m1_secondary_always_eager),
                    "lacks actual replay coverage",
                ),
            ]
        )

        for name, mutation, marker in mutations:
            case = Path(temporary) / name
            shutil.copytree(base, case)
            mutation(case)
            try:
                validate_artifact(
                    case,
                    expected_source,
                    allow_internal_fixture=True,
                )
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
    parser.add_argument(
        "--scope",
        choices=sorted(SCOPE_CONTRACTS),
        default=RELEASE_SCOPE,
    )
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
        summary = validate_artifact(
            args.artifact_root,
            current_git_state(),
            args.scope,
        )
        manifest = validation_manifest(args.artifact_root, out_dir, summary)
        write_json(out_dir / "manifest.json", manifest, exclusive=True)
        print(manifest["pass_line"])
        return 0
    except (DeterminismArtifactError, OSError) as error:
        write_rejection_manifest(args.artifact_root, out_dir, error, args.scope)
        print(
            f"{scope_contract(args.scope)['pass_prefix'].removesuffix(' PASS')} REJECT: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
