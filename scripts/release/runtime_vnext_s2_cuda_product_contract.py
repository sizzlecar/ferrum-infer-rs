#!/usr/bin/env python3
"""Build and verify the focused Qwen3.5-4B CUDA S2 product contract."""

from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import runtime_vnext_cuda_determinism as determinism  # noqa: E402
import runtime_vnext_g01_core_contracts as g01  # noqa: E402
import runtime_vnext_g02_core as g02  # noqa: E402
import runtime_vnext_s1_cuda_capacity as s1_capacity  # noqa: E402
import runtime_vnext_s1_cuda_checkpoint as s1  # noqa: E402
import runtime_vnext_s1_cuda_decode_capacity as s1_decode_capacity  # noqa: E402
import runtime_vnext_s2_api_modality_checkpoint as api_modality  # noqa: E402
import runtime_vnext_s2_historical_resource_source as historical  # noqa: E402
import runtime_vnext_s2_latency_failure_checkpoint as latency_failure  # noqa: E402
import runtime_vnext_s2_multiturn_concurrency_checkpoint as multiturn  # noqa: E402
import runtime_vnext_s2_response_format_checkpoint as response_format  # noqa: E402
import runtime_vnext_s2_stream_disconnect_checkpoint as stream_disconnect  # noqa: E402
import runtime_vnext_s2_tool_schema_checkpoint as tool_schema  # noqa: E402


PASS_PREFIX = "FERRUM RUNTIME VNEXT S2 CUDA PRODUCT CONTRACT PASS"
SELFTEST_PASS = "FERRUM RUNTIME VNEXT S2 CUDA PRODUCT CONTRACT SELFTEST PASS"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION_RE = re.compile(
    r"(?:^|/)models--Qwen--Qwen3\.5-4B/snapshots/([0-9a-f]{40})(?=/|\s|$)"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DEPENDENCY_CLOSURE_FILES = frozenset(
    {
        "scripts/release/runtime_vnext_r0_core_closure.py",
        "scripts/release/runtime_vnext_s2_cuda_product_contract.py",
    }
)


@dataclass(frozen=True)
class InputSpec:
    lane: str
    child_kind: str
    child_suffix: str
    pass_prefix: str
    identity_field: str
    identity_value: str
    raw_field: str | None
    validator: str
    product: bool = False


INPUT_SPECS: dict[str, InputSpec] = {
    "g01": InputSpec(
        "vnext-g01",
        "vnext-g01",
        "g01-contracts/manifest.json",
        "FERRUM RUNTIME VNEXT G01 CORE CONTRACTS PASS",
        "artifact_type",
        "runtime_vnext_g01_core_contracts_manifest",
        None,
        "g01",
    ),
    "s1": InputSpec(
        "vnext-s1-cuda",
        "delegated-manifest",
        "manifest.json",
        "FERRUM RUNTIME VNEXT S1 CUDA BASIC SLICE PASS",
        "artifact_type",
        "runtime_vnext_s1_cuda_basic_slice_manifest",
        "raw_artifact_dir",
        "s1",
        True,
    ),
    "s1_capacity": InputSpec(
        "vnext-s1-cuda-capacity",
        "delegated-manifest",
        "manifest.json",
        "FERRUM RUNTIME VNEXT S1 CUDA CAPACITY PRESSURE PASS",
        "artifact_type",
        "runtime_vnext_s1_cuda_capacity_pressure_validation_v2",
        "source_artifact",
        "s1_capacity",
        True,
    ),
    "s1_decode_capacity": InputSpec(
        "vnext-s1-cuda-decode-capacity",
        "delegated-manifest",
        "manifest.json",
        "FERRUM RUNTIME VNEXT S1 CUDA DECODE CAPACITY PASS",
        "artifact_type",
        "runtime_vnext_s1_cuda_decode_capacity_validation",
        "source_artifact",
        "s1_decode_capacity",
        True,
    ),
    "g02_core": InputSpec(
        "vnext-g02-core",
        "delegated-manifest",
        "g02-core/manifest.json",
        "FERRUM RUNTIME VNEXT G02 CORE L0 L1 PASS",
        "schema",
        g02.SCHEMA,
        None,
        "g02_core",
    ),
    "m1_determinism": InputSpec(
        "vnext-s2-m1-determinism",
        "delegated-manifest",
        "manifest.json",
        "FERRUM RUNTIME VNEXT M1 S2 CUDA DETERMINISM FOCUSED PASS",
        "artifact_type",
        determinism.VALIDATOR_ARTIFACT_TYPE,
        "input_artifact_root",
        "m1_determinism",
        True,
    ),
    "response_format": InputSpec(
        "vnext-s2-response-format",
        "delegated-manifest",
        "manifest.json",
        response_format.PASS_PREFIX,
        "checkpoint_id",
        response_format.CHECKPOINT_ID,
        "source_root",
        "response_format",
        True,
    ),
    "api_modality": InputSpec(
        "vnext-s2-api-modality",
        "delegated-manifest",
        "manifest.json",
        api_modality.PASS_PREFIX,
        "checkpoint_id",
        api_modality.CHECKPOINT_ID,
        "source_root",
        "api_modality",
        True,
    ),
    "stream_disconnect": InputSpec(
        "vnext-s2-stream-disconnect",
        "delegated-manifest",
        "manifest.json",
        stream_disconnect.PASS_PREFIX,
        "checkpoint_id",
        stream_disconnect.CHECKPOINT_ID,
        "source_root",
        "stream_disconnect",
        True,
    ),
    "tool_schema": InputSpec(
        "vnext-s2-tool-schema",
        "delegated-manifest",
        "manifest.json",
        tool_schema.PASS_PREFIX,
        "checkpoint_id",
        tool_schema.CHECKPOINT_ID,
        "source_root",
        "tool_schema",
        True,
    ),
    "multiturn_concurrency": InputSpec(
        "vnext-s2-multiturn-concurrency",
        "delegated-manifest",
        "manifest.json",
        multiturn.PASS_PREFIX,
        "checkpoint_id",
        multiturn.CHECKPOINT_ID,
        "source_root",
        "multiturn_concurrency",
        True,
    ),
    "latency_first_failure": InputSpec(
        "vnext-s2-latency-first-failure",
        "delegated-manifest",
        "manifest.json",
        latency_failure.PASS_PREFIX,
        "checkpoint_id",
        latency_failure.CHECKPOINT_ID,
        "source_root",
        "latency_first_failure",
        True,
    ),
    "historical_resource_source": InputSpec(
        "vnext-s2-historical-resource-source",
        "delegated-manifest",
        "manifest.json",
        historical.PASS_PREFIX,
        "checkpoint_id",
        historical.CHECKPOINT_ID,
        None,
        "historical_resource_source",
    ),
}

SCENARIO_KEYS = {
    "response_format",
    "api_modality",
    "stream_disconnect",
    "tool_schema",
    "multiturn_concurrency",
}
PRODUCT_KEYS = {key for key, spec in INPUT_SPECS.items() if spec.product}
ACCEPTANCE = {
    "all_required_children_present": True,
    "all_children_revalidated_from_raw_evidence": True,
    "single_clean_source_identity": True,
    "single_cuda_binary_identity": True,
    "qwen35_4b_revision_bound": True,
    "model_file_closure_bound": True,
    "typed_cuda_configs_bound": True,
    "runner_and_validator_sources_bound": True,
    "single_rtx4090_hardware_class": True,
    "run_and_serve_product_paths_covered": True,
    "run_and_serve_same_resolved_execution_plan": True,
    "run_and_serve_same_runtime_implementation": True,
    "production_legacy_selection_zero": True,
    "focused_scope_does_not_claim_full_g02_or_release": True,
}
DOES_NOT_PROVE = [
    "full G02",
    "full G04",
    "full G05",
    "full G06",
    "full G08 model matrix",
    "Metal",
    "formal performance",
    "release readiness",
]


class AggregateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateError(message)


def require_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be an array")
    return value


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be non-empty")
    return value


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {item}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise AggregateError(f"invalid {label}: {path}: {error}") from error
    return require_object(value, label)


def read_text(path: Path, label: str) -> str:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    try:
        value = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise AggregateError(f"invalid {label}: {path}: {error}") from error
    require("\x00" not in value and "\ufffd" not in value, f"{label} contains invalid text")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"cannot hash non-regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def clean_source() -> dict[str, Any]:
    status = [line for line in git("status", "--short", "--untracked-files=all").splitlines() if line]
    require(not status, f"S2 requires a clean checkout: {status}")
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_tree_sha": git("rev-parse", "HEAD^{tree}"),
        "dirty": False,
        "status_short": [],
    }


def source_for_sha(git_sha: str) -> dict[str, Any]:
    require(GIT_SHA_RE.fullmatch(git_sha) is not None, "source git SHA is invalid")
    tree = git("show", "-s", "--format=%T", git_sha)
    require(GIT_SHA_RE.fullmatch(tree) is not None, "source git tree SHA is invalid")
    return {
        "git_sha": git_sha,
        "git_tree_sha": tree,
        "dirty": False,
        "status_short": [],
    }


def dependency_control_plane_only(paths: list[str]) -> tuple[list[str], list[str]]:
    allowed = [path for path in paths if path in DEPENDENCY_CLOSURE_FILES]
    rejected = [path for path in paths if path not in DEPENDENCY_CLOSURE_FILES]
    return allowed, rejected


def dependency_source_closure(
    dependency_source: dict[str, Any], current_source: dict[str, Any]
) -> dict[str, Any]:
    recorded = source_for_sha(
        require_string(dependency_source.get("git_sha"), "S2 dependency git SHA")
    )
    require(recorded == dependency_source, "S2 dependency source differs from git")
    current = source_for_sha(
        require_string(current_source.get("git_sha"), "S2 current git SHA")
    )
    require(current == current_source, "S2 current source differs from git")
    git(
        "merge-base",
        "--is-ancestor",
        recorded["git_sha"],
        current["git_sha"],
    )
    changed = [
        line
        for line in git(
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            f"{recorded['git_sha']}..{current['git_sha']}",
        ).splitlines()
        if line
    ]
    allowed, rejected = dependency_control_plane_only(changed)
    require(
        not rejected,
        "S2 evidence is stale after product, scenario, or validator changes: "
        f"{rejected[:8]}",
    )
    return {
        "from_git_sha": recorded["git_sha"],
        "to_git_sha": current["git_sha"],
        "changed_files": allowed,
        "changed_file_count": len(allowed),
        "policy": "s2-aggregate-control-plane-only",
    }


def safe_relative_path(root: Path, relative: str, label: str) -> Path:
    value = Path(relative)
    require(not value.is_absolute() and ".." not in value.parts, f"{label} is unsafe")
    candidate = (root / value).resolve()
    require(root.resolve() in candidate.parents, f"{label} escapes its root")
    require(candidate.is_file() and not candidate.is_symlink(), f"{label} is missing: {candidate}")
    return candidate


def relocate_recorded_path(
    recorded: str,
    *,
    outer_path: Path,
    recorded_outer_dir: str,
    label: str,
    directory: bool = False,
) -> Path:
    recorded_path = Path(recorded).expanduser()
    if recorded_path.exists() and not recorded_path.is_symlink():
        resolved = recorded_path.resolve()
    else:
        recorded_outer = Path(recorded_outer_dir)
        actual_outer = outer_path.parent.resolve()
        candidates: list[Path] = []
        try:
            candidates.append(actual_outer / recorded_path.relative_to(recorded_outer))
        except ValueError:
            pass
        try:
            candidates.append(actual_outer.parent / recorded_path.relative_to(recorded_outer.parent))
        except ValueError:
            pass
        matches = sorted(
            {
                candidate.resolve()
                for candidate in candidates
                if candidate.exists() and not candidate.is_symlink()
            }
        )
        require(len(matches) == 1, f"cannot relocate {label}: {recorded}")
        resolved = matches[0]
    if directory:
        require(resolved.is_dir(), f"{label} is not a directory: {resolved}")
    else:
        require(resolved.is_file(), f"{label} is not a file: {resolved}")
    return resolved


def normalize_paths(value: Any, actual: Path, recorded: str) -> Any:
    if isinstance(value, dict):
        return {key: normalize_paths(item, actual, recorded) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_paths(item, actual, recorded) for item in value]
    if isinstance(value, str):
        actual_text = str(actual)
        if value == actual_text or value.startswith(actual_text + "/"):
            return recorded + value[len(actual_text) :]
    return value


def child_source_sha(child: dict[str, Any], key: str) -> str:
    candidates: list[Any] = [
        child.get("source_git_sha"),
        child.get("source", {}).get("git_sha") if isinstance(child.get("source"), dict) else None,
        child.get("source_identity", {}).get("git_sha")
        if isinstance(child.get("source_identity"), dict)
        else None,
        child.get("evidence", {}).get("git_sha")
        if isinstance(child.get("evidence"), dict)
        else None,
    ]
    values = {value for value in candidates if isinstance(value, str) and value}
    require(len(values) == 1, f"{key} child source SHA is missing or ambiguous: {sorted(values)}")
    value = next(iter(values))
    require(GIT_SHA_RE.fullmatch(value) is not None, f"{key} child source SHA is invalid")
    return value


def validate_outer_child_pair(
    key: str,
    outer: dict[str, Any],
    child: dict[str, Any],
    child_digest: str,
) -> dict[str, Any]:
    spec = INPUT_SPECS[key]
    require(outer.get("schema_version") == 1, f"{key} outer schema mismatch")
    require(outer.get("lane") == spec.lane and outer.get("status") == "pass", f"{key} outer lane/status mismatch")
    git_sha = require_string(outer.get("git_sha"), f"{key} outer git_sha")
    dirty = require_object(outer.get("dirty_status"), f"{key} outer dirty_status")
    require(dirty == {"is_dirty": False, "status_short": []}, f"{key} outer source was dirty")
    artifact_dir = require_string(outer.get("artifact_dir"), f"{key} outer artifact_dir")
    require(
        outer.get("pass_line") == f"FERRUM GATE {spec.lane} PASS: {artifact_dir}",
        f"{key} outer PASS line mismatch",
    )
    artifacts = require_object(outer.get("child_artifacts"), f"{key} child_artifacts")
    require(artifacts.get("kind") == spec.child_kind, f"{key} child kind mismatch")
    child_ref = require_object(artifacts.get("child_manifest"), f"{key} child reference")
    require(child_ref.get("sha256") == child_digest, f"{key} outer/child SHA256 mismatch")
    require(child.get("status") == "pass", f"{key} child did not pass")
    require(child.get(spec.identity_field) == spec.identity_value, f"{key} child identity mismatch")
    child_pass = require_string(child.get("pass_line"), f"{key} child pass_line")
    require(child_pass.startswith(spec.pass_prefix + ": "), f"{key} child PASS prefix mismatch")
    require(outer.get("child_pass_line") == child_pass, f"{key} outer/child PASS mismatch")
    source_sha = child_source_sha(child, key)
    require(source_sha == git_sha, f"{key} outer/child source SHA mismatch")
    return source_for_sha(source_sha)


def find_model_revisions(value: Any) -> set[str]:
    revisions: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            revisions.update(find_model_revisions(item))
    elif isinstance(value, list):
        for item in value:
            revisions.update(find_model_revisions(item))
    elif isinstance(value, str):
        revisions.update(MODEL_REVISION_RE.findall(value))
    return revisions


def model_revision_binding(
    key: str,
    raw: Path,
    documents: list[Any],
) -> dict[str, Any]:
    revisions: set[str] = set()
    for document in documents:
        revisions.update(find_model_revisions(document))

    server_log = raw / "server.log"
    log_sha256 = None
    if server_log.is_file() and not server_log.is_symlink():
        revisions.update(
            find_model_revisions(read_text(server_log, f"{key} server log"))
        )
        log_sha256 = sha256(server_log)

    require(
        len(revisions) == 1,
        f"{key} must bind one Qwen3.5-4B revision: {sorted(revisions)}",
    )
    return {
        "model_id": MODEL_ID,
        "model_revision": next(iter(revisions)),
        "resolution_log_sha256": log_sha256,
    }


def hardware_from_value(value: Any) -> dict[str, Any] | None:
    rows: list[str] = []
    if isinstance(value, str):
        rows = [line.strip() for line in value.splitlines() if line.strip()]
    elif isinstance(value, list):
        rows = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, dict):
        if isinstance(value.get("stdout"), str):
            rows = [line.strip() for line in value["stdout"].splitlines() if line.strip()]
        else:
            name = value.get("name") or value.get("device_name")
            if isinstance(name, str):
                return {
                    "device_count": 1,
                    "device_name": name,
                    "uuid": value.get("uuid"),
                    "memory_bytes": value.get("memory_bytes"),
                    "memory_mib": value.get("memory_total_mib"),
                    "driver_version": value.get("driver_version"),
                }
    if not rows:
        return None
    require(len(rows) == 1, f"hardware evidence must contain one GPU row: {rows}")
    parts = [part.strip() for part in rows[0].split(",")]
    name = next((part for part in parts if "4090" in part), rows[0])
    uuid = next((part for part in parts if part.startswith("GPU-")), None)
    numeric = []
    for part in parts:
        match = re.fullmatch(r"\s*([0-9]+)(?:\s+MiB)?\s*", part)
        if match is not None:
            numeric.append(int(match.group(1)))
    memory_mib = next((item for item in numeric if 20_000 <= item <= 30_000), None)
    driver = next((part for part in reversed(parts) if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", part)), None)
    return {
        "device_count": 1,
        "device_name": name,
        "uuid": uuid,
        "memory_bytes": None,
        "memory_mib": memory_mib,
        "driver_version": driver,
    }


def normalized_hardware_class(value: dict[str, Any]) -> dict[str, Any]:
    memory_bytes = value.get("memory_bytes")
    memory_mib = value.get("memory_mib")
    if isinstance(memory_bytes, int):
        memory_mib = round(memory_bytes / (1024 * 1024))
    require(value.get("device_count") == 1, "hardware evidence is not single-GPU")
    name = str(value.get("device_name") or "")
    require("4090" in name, f"hardware evidence is not RTX 4090: {name}")
    require(isinstance(memory_mib, int) and 20_000 <= memory_mib <= 30_000, "hardware memory evidence is invalid")
    return {
        "device_count": 1,
        "device_name": "NVIDIA GeForce RTX 4090",
        "memory_mib": memory_mib,
        "driver_version": value.get("driver_version"),
        "uuid": value.get("uuid"),
    }


def scenario_receipt(raw: Path) -> dict[str, Any]:
    return read_json(raw / "execution_receipt.json", "scenario execution receipt")


def product_binding(
    key: str,
    raw: Path,
    evidence: dict[str, Any],
    validator_path: Path,
) -> dict[str, Any]:
    documents: list[Any] = [evidence]
    for filename in ("summary.json", "execution_receipt.json", "collection.json", "provenance.json"):
        path = raw / filename
        if path.is_file() and not path.is_symlink():
            documents.append(read_json(path, f"{key} {filename}"))
    model = model_revision_binding(key, raw, documents)

    binary_values: set[str] = set()
    for document in documents:
        if isinstance(document, dict):
            for candidate in (
                document.get("binary_sha256"),
                document.get("execution_receipt", {}).get("binary_sha256")
                if isinstance(document.get("execution_receipt"), dict)
                else None,
            ):
                if isinstance(candidate, str) and SHA256_RE.fullmatch(candidate):
                    binary_values.add(candidate)
    require(len(binary_values) == 1, f"{key} binary identity is missing or ambiguous")

    hardware_candidates: list[dict[str, Any]] = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        for candidate in (
            document.get("hardware"),
            document.get("execution_receipt", {}).get("hardware")
            if isinstance(document.get("execution_receipt"), dict)
            else None,
        ):
            parsed = hardware_from_value(candidate)
            if parsed is not None:
                hardware_candidates.append(normalized_hardware_class(parsed))
    require(hardware_candidates, f"{key} hardware identity is missing")
    hardware = hardware_candidates[0]
    for candidate in hardware_candidates[1:]:
        for field in ("device_count", "device_name", "memory_mib"):
            require(candidate[field] == hardware[field], f"{key} hardware {field} is inconsistent")
        for field in ("uuid", "driver_version"):
            if candidate.get(field) and hardware.get(field):
                require(candidate[field] == hardware[field], f"{key} hardware {field} is inconsistent")
            elif candidate.get(field):
                hardware[field] = candidate[field]

    runner_sha = None
    manifest_sha = None
    effective_configs: list[str] = []
    if key in SCENARIO_KEYS:
        receipt = scenario_receipt(raw)
        runner_sha = require_string(receipt.get("runner_sha256"), f"{key} runner_sha256")
        manifest_sha = require_string(receipt.get("manifest_sha256"), f"{key} manifest_sha256")
        require(SHA256_RE.fullmatch(runner_sha) is not None, f"{key} runner SHA256 is invalid")
        require(SHA256_RE.fullmatch(manifest_sha) is not None, f"{key} manifest SHA256 is invalid")
        for path in raw.rglob("*effective_config*.json"):
            if path.is_file() and not path.is_symlink():
                effective_configs.append(sha256(path))
        require(effective_configs, f"{key} has no typed effective config evidence")

    binding = {
        "binary_sha256": next(iter(binary_values)),
        **model,
        "hardware": hardware,
        "runner_sha256": runner_sha,
        "scenario_manifest_sha256": manifest_sha,
        "effective_config_sha256": sorted(set(effective_configs)),
        "validator": {
            "path": validator_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256(validator_path),
        },
    }
    if key == "multiturn_concurrency":
        observability = require_object(
            evidence.get("observability"),
            "multiturn_concurrency observability",
        )
        identity = require_object(
            observability.get("product_execution_identity"),
            "multiturn_concurrency product execution identity",
        )
        require(
            identity.get("entrypoints") == ["run", "serve"]
            and identity.get("same_resolved_execution_plan") is True
            and identity.get("same_runtime_implementation") is True
            and identity.get("production_legacy_selection_count") == 0,
            "multiturn_concurrency product execution identity failed",
        )
        for field in (
            "resolved_execution_plan_hash",
            "runtime_implementation_fingerprint",
        ):
            require(
                SHA256_RE.fullmatch(str(identity.get(field))) is not None,
                f"multiturn_concurrency {field} is invalid",
            )
        binding["product_execution_identity"] = copy.deepcopy(identity)
    return binding


def require_evidence_match(
    key: str,
    child: dict[str, Any],
    observed: dict[str, Any],
    raw: Path,
    recorded_raw: str,
) -> None:
    recorded = require_object(child.get("evidence"), f"{key} recorded evidence")
    normalized = normalize_paths(observed, raw, recorded_raw)
    require(recorded == normalized, f"{key} child evidence differs from raw revalidation")


def test_evidence_passed(value: Any, label: str) -> int:
    evidence = require_object(value, f"{label} test evidence")
    summary = require_object(evidence.get("summary"), f"{label} test summary")
    passed = summary.get("passed")
    require(type(passed) is int and passed >= 0, f"{label} passed count is invalid")
    return passed


def validate_g02_at_source(
    child_path: Path, source: dict[str, Any]
) -> dict[str, Any]:
    original_git = g02.git

    def source_git(*args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return require_string(source.get("git_sha"), "G02 source git SHA")
        if args == ("rev-parse", "HEAD^{tree}"):
            return require_string(source.get("git_tree_sha"), "G02 source git tree")
        return original_git(*args)

    g02.git = source_git
    try:
        return g02.validate_artifact(child_path.parent)
    finally:
        g02.git = original_git


def validate_s1(
    key: str,
    child: dict[str, Any],
    child_path: Path,
    raw: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    correctness = s1.validate(raw, source["git_sha"])
    s1.require_bounded_overhead_native_evidence(correctness)
    performance = s1.validate_profile_overhead(raw)
    product = s1.validate_product_commands(raw)
    validation_path = child_path.parent / "validation.json"
    validation_ref = require_object(child.get("validation"), "S1 validation reference")
    require(
        validation_ref.get("sha256") == sha256(validation_path)
        and validation_ref.get("size_bytes") == validation_path.stat().st_size,
        "S1 validation reference mismatch",
    )
    validation = read_json(validation_path, "S1 validation")
    require(
        validation.get("source_git_sha") == source["git_sha"]
        and validation.get("binary_sha256") == correctness["binary_sha256"]
        and validation.get("correctness")
        == {"run": correctness["run"], "serve": correctness["serve"]}
        and validation.get("profile_overhead") == performance
        and validation.get("product") == product,
        "S1 persisted validation differs from raw revalidation",
    )
    require(
        validation.get("raw_artifact_index") == s1.artifact_index(raw)
        and validation.get("raw_artifact_index_sha256")
        == s1.canonical_json_sha256(validation["raw_artifact_index"]),
        "S1 raw artifact index mismatch",
    )
    evidence = {
        "binary_sha256": correctness["binary_sha256"],
        "model": product.get("model_snapshot_path"),
        "hardware": correctness["hardware"],
    }
    return product_binding(key, raw, evidence, Path(s1.__file__).resolve())


def validate_capacity_child(
    key: str,
    child: dict[str, Any],
    raw: Path,
    validator: Callable[[Path, Path], int],
    validator_path: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"ferrum-s2-{key}-") as temporary:
        out = Path(temporary) / "validation"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            returncode = validator(raw, out)
        require(
            returncode == 0,
            f"{key} raw revalidation failed: {stderr.getvalue().strip()}",
        )
        observed = read_json(out / "manifest.json", f"{key} regenerated manifest")
    for field in (
        "artifact_type",
        "status",
        "source_git_sha",
        "binary_sha256",
        "model_path",
        "source_collection_sha256",
    ):
        require(child.get(field) == observed.get(field), f"{key} persisted {field} mismatch")
    evidence = {
        "binary_sha256": observed["binary_sha256"],
        "model": observed["model_path"],
    }
    provenance_path = raw / "provenance.json"
    if provenance_path.is_file():
        provenance = read_json(provenance_path, f"{key} provenance")
        evidence["hardware"] = provenance.get("nvidia_smi")
    return product_binding(key, raw, evidence, validator_path)


def validate_historical_output(child_path: Path, child: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    root = child_path.parent
    require(child.get("scope") == list(historical.CASE_IDS), "historical focused case denominator mismatch")
    require(child.get("full_s2") is False and child.get("product_evidence_complete") is False, "historical source lane overclaims S2")
    identity = require_object(child.get("source_identity"), "historical source identity")
    require(
        identity.get("git_sha") == source["git_sha"]
        and identity.get("git_tree_sha") == source["git_tree_sha"]
        and identity.get("dirty_status")
        == {"is_dirty": False, "status_short": []},
        "historical source identity mismatch",
    )
    config = historical.validate_config()
    require(child.get("product_evidence_requirements") == historical.EXPECTED_PRODUCT_EVIDENCE, "historical product evidence matrix mismatch")
    inputs = require_object(child.get("inputs"), "historical inputs")
    for name, reference in inputs.items():
        ref = require_object(reference, f"historical input {name}")
        path = safe_relative_path(REPO_ROOT, require_string(ref.get("path"), f"historical input {name}.path"), f"historical input {name}")
        require(ref.get("sha256") == sha256(path), f"historical input {name} is stale")
    tree = read_json(root / "artifact_tree.json", "historical artifact tree")
    tree_rows = require_list(tree.get("files"), "historical artifact tree files")
    require(tree.get("schema_version") == 1 and tree.get("file_count") == len(tree_rows), "historical artifact tree header mismatch")
    recorded_tree_paths: set[str] = set()
    for raw_ref in tree_rows:
        ref = require_object(raw_ref, "historical artifact tree file")
        relative = require_string(ref.get("path"), "historical artifact tree file.path")
        require(relative not in recorded_tree_paths, f"duplicate historical artifact tree path: {relative}")
        recorded_tree_paths.add(relative)
        member = safe_relative_path(root, relative, f"historical artifact tree {relative}")
        require(
            ref.get("sha256") == sha256(member)
            and ref.get("size_bytes") == member.stat().st_size,
            f"historical artifact tree member mismatch: {relative}",
        )
    wrapper_files = {
        "gate.manifest.json",
        "run_gate.child.command.json",
        "run_gate.child.stdout",
        "run_gate.child.stderr",
    }
    actual_tree_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name != "artifact_tree.json"
        and path.relative_to(root).as_posix() not in wrapper_files
    }
    require(actual_tree_paths == recorded_tree_paths, "historical artifact tree file set mismatch")

    resource = require_object(child.get("resource_invariants"), "historical resource invariants")
    for name in ("manifest", "report", "runner_stdout", "runner_stderr"):
        ref = require_object(resource.get(name), f"historical resource {name}")
        path = safe_relative_path(root, require_string(ref.get("path"), f"historical resource {name}.path"), f"historical resource {name}")
        require(ref.get("sha256") == sha256(path) and ref.get("size_bytes") == path.stat().st_size, f"historical resource {name} reference mismatch")
    resource_manifest = read_json(
        safe_relative_path(root, resource["manifest"]["path"], "historical resource manifest"),
        "historical resource manifest",
    )
    resource_report = read_json(
        safe_relative_path(root, resource["report"]["path"], "historical resource report"),
        "historical resource report",
    )
    require(
        resource_manifest.get("status") == "pass"
        and resource_manifest.get("git_sha") == source["git_sha"]
        and resource_manifest.get("git_dirty") is False,
        "historical resource invariant provenance mismatch",
    )
    require(
        resource_report.get("status") == "pass"
        and all(resource_report.get(field) == 0 for field in ("leaked_resources", "underflow_count", "silent_oom_count", "panic_count")),
        "historical resource invariant report failed",
    )

    tests = require_list(child.get("source_tests"), "historical source tests")
    require(len(tests) == len(config["tests"]) == 7, "historical source test denominator mismatch")
    for row in tests:
        test = require_object(row, "historical source test")
        expected = config["tests"].get(test.get("id"))
        require(isinstance(expected, dict), f"unknown historical source test: {test.get('id')}")
        source_ref = require_object(test.get("source"), f"historical {test.get('id')} source")
        source_path = safe_relative_path(REPO_ROOT, source_ref["path"], f"historical {test.get('id')} source")
        require(source_ref.get("sha256") == sha256(source_path), f"historical source test is stale: {test.get('id')}")
        for name in ("receipt", "stdout", "stderr", "runner_stdout", "runner_stderr"):
            ref = require_object(test.get(name), f"historical {test.get('id')} {name}")
            path = safe_relative_path(root, ref["path"], f"historical {test.get('id')} {name}")
            require(ref.get("sha256") == sha256(path) and ref.get("size_bytes") == path.stat().st_size, f"historical {test.get('id')} {name} mismatch")
        receipt = read_json(safe_relative_path(root, test["receipt"]["path"], "historical bounded receipt"), "historical bounded receipt")
        require(
            receipt.get("status") == "pass"
            and receipt.get("reason") == "command_completed"
            and receipt.get("rc") == 0
            and receipt.get("cleanup") == {"process_group_gone": True},
            f"historical bounded receipt failed: {test.get('id')}",
        )
        stdout = safe_relative_path(root, test["stdout"]["path"], "historical source stdout").read_text(encoding="utf-8")
        require(stdout.splitlines().count(f"test {test['test_name']} ... ok") == 1, f"historical source test output mismatch: {test.get('id')}")

    cases = require_list(child.get("cases"), "historical cases")
    require([row.get("id") for row in cases if isinstance(row, dict)] == list(historical.CASE_IDS), "historical case order mismatch")
    for row in cases:
        case = require_object(row, "historical case")
        replay = require_object(case.get("historical_replay"), f"historical replay {case.get('id')}")
        require(replay.get("returncode") == historical.EXPECTED_REPLAY_RC, f"historical replay did not kill {case.get('id')}")
        for name in ("evidence", "input", "mutation", "failure_log", "current_replay_stdout", "current_replay_stderr"):
            ref = require_object(replay.get(name), f"historical replay {case.get('id')} {name}")
            path = safe_relative_path(root, ref["path"], f"historical replay {case.get('id')} {name}")
            require(ref.get("sha256") == sha256(path) and ref.get("size_bytes") == path.stat().st_size, f"historical replay {case.get('id')} {name} mismatch")
        stdout = safe_relative_path(root, replay["current_replay_stdout"]["path"], "historical replay stdout").read_text(encoding="utf-8")
        require(stdout.splitlines().count(replay["failure_signature"]) == 1, f"historical replay signature mismatch: {case.get('id')}")
    return {
        "validator": {
            "path": Path(historical.__file__).resolve().relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256(Path(historical.__file__).resolve()),
        },
        "case_count": len(cases),
        "source_test_count": len(tests),
        "artifact_tree_sha256": sha256(root / "artifact_tree.json"),
    }


def deep_validate(
    key: str,
    child: dict[str, Any],
    child_path: Path,
    outer_path: Path,
    outer: dict[str, Any],
    source: dict[str, Any],
    *,
    verify_checkout: bool,
) -> tuple[dict[str, Any], Path | None]:
    spec = INPUT_SPECS[key]
    if spec.validator == "g01":
        summary = g01.verify_checkpoint_manifest(
            child_path, verify_checkout=verify_checkout
        )
        require(summary.get("source") == source, "G01 deep source binding mismatch")
        validator_path = Path(g01.__file__).resolve()
        return {
            "validator": {
                "path": validator_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256(validator_path),
            }
        }, None
    if spec.validator == "g02_core":
        observed = validate_g02_at_source(child_path, source)
        require(observed.get("source") == source, "G02 core source binding mismatch")
        validator_path = Path(g02.__file__).resolve()
        return {
            "validator": {
                "path": validator_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256(validator_path),
            },
            "l0_test_count": observed["l0"]["test_count"],
            "l1_test_count": test_evidence_passed(
                observed["l1"].get("test_evidence"), "G02 L1"
            ),
        }, None
    if spec.validator == "historical_resource_source":
        return validate_historical_output(child_path, child, source), None

    require(spec.raw_field is not None, f"{key} has no raw artifact field")
    recorded_raw = require_string(child.get(spec.raw_field), f"{key} {spec.raw_field}")
    raw = relocate_recorded_path(
        recorded_raw,
        outer_path=outer_path,
        recorded_outer_dir=require_string(outer.get("artifact_dir"), f"{key} outer artifact_dir"),
        label=f"{key} raw artifact",
        directory=True,
    )

    try:
        if spec.validator == "s1":
            binding = validate_s1(key, child, child_path, raw, source)
        elif spec.validator == "s1_capacity":
            binding = validate_capacity_child(
                key,
                child,
                raw,
                s1_capacity.validate,
                Path(s1_capacity.__file__).resolve(),
            )
        elif spec.validator == "s1_decode_capacity":
            binding = validate_capacity_child(
                key,
                child,
                raw,
                s1_decode_capacity.validate,
                Path(s1_decode_capacity.__file__).resolve(),
            )
        elif spec.validator == "m1_determinism":
            observed = determinism.validate_artifact(
                raw,
                {"git_sha": source["git_sha"], "git_tree_sha": source["git_tree_sha"]},
                "m1-s2-focused",
            )
            for field, value in observed.items():
                require(child.get(field) == value, f"M1 determinism persisted {field} mismatch")
            probe = read_json(raw / "hardware-probe/probe.json", "M1 determinism hardware probe")
            normalized = require_object(probe.get("normalized"), "M1 determinism normalized hardware")
            hardware = normalized_hardware_class(
                {
                    "device_count": normalized.get("device_count"),
                    "device_name": normalized.get("device_name"),
                    "memory_bytes": normalized.get("memory_bytes"),
                    "memory_mib": None,
                    "driver_version": normalized.get("runtime", {}).get("driver_version")
                    if isinstance(normalized.get("runtime"), dict)
                    else None,
                    "uuid": None,
                }
            )
            models_lock = read_json(raw / "models.lock.json", "M1 determinism models lock")
            revisions = find_model_revisions(models_lock)
            require(len(revisions) == 1, "M1 determinism model revision binding mismatch")
            binding = {
                "binary_sha256": observed["binary_sha256"],
                "model_id": MODEL_ID,
                "model_revision": next(iter(revisions)),
                "hardware": hardware,
                "hardware_id": probe.get("hardware_id"),
                "device_fingerprint": observed["device_fingerprint"],
                "runner_sha256": None,
                "scenario_manifest_sha256": None,
                "effective_config_sha256": [],
                "model_file_closure_sha256": canonical_sha256(models_lock),
                "native_operator_set_lock_sha256": read_json(raw / "evidence.json", "M1 determinism evidence")["source"]["native_operator_set_lock"]["sha256"],
                "validator": {
                    "path": Path(determinism.__file__).resolve().relative_to(REPO_ROOT).as_posix(),
                    "sha256": sha256(Path(determinism.__file__).resolve()),
                },
            }
        else:
            modules: dict[str, tuple[Any, Path]] = {
                "response_format": (response_format, Path(response_format.__file__).resolve()),
                "api_modality": (api_modality, Path(api_modality.__file__).resolve()),
                "stream_disconnect": (stream_disconnect, Path(stream_disconnect.__file__).resolve()),
                "tool_schema": (tool_schema, Path(tool_schema.__file__).resolve()),
                "multiturn_concurrency": (multiturn, Path(multiturn.__file__).resolve()),
                "latency_first_failure": (latency_failure, Path(latency_failure.__file__).resolve()),
            }
            module, validator_path = modules[spec.validator]
            observed = module.validate_source(raw, source["git_sha"])
            require_evidence_match(key, child, observed, raw, recorded_raw)
            binding = product_binding(key, raw, observed, validator_path)
            if key == "latency_first_failure":
                model = require_object(observed.get("model"), "latency model closure")
                binding["model_file_closure_sha256"] = require_string(
                    model.get("closure_sha256"), "latency model closure SHA256"
                )
    except AggregateError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise AggregateError(f"{key} raw artifact revalidation failed: {error}") from error
    return binding, raw


def load_input(path: Path, key: str, *, verify_checkout: bool) -> dict[str, Any]:
    outer_path = path.expanduser().resolve()
    outer = read_json(outer_path, f"{key} outer manifest")
    recorded_outer_dir = require_string(outer.get("artifact_dir"), f"{key} outer artifact_dir")
    artifacts = require_object(outer.get("child_artifacts"), f"{key} child artifacts")
    child_ref = require_object(artifacts.get("child_manifest"), f"{key} child reference")
    child_recorded = require_string(child_ref.get("path"), f"{key} child path")
    child_path = relocate_recorded_path(
        child_recorded,
        outer_path=outer_path,
        recorded_outer_dir=recorded_outer_dir,
        label=f"{key} child manifest",
    )
    require(
        child_path.as_posix().endswith("/" + INPUT_SPECS[key].child_suffix),
        f"{key} child manifest layout mismatch: {child_path}",
    )
    child_digest = sha256(child_path)
    child = read_json(child_path, f"{key} child manifest")
    source = validate_outer_child_pair(key, outer, child, child_digest)
    if verify_checkout:
        require(source == clean_source(), f"{key} is stale against current checkout")
    binding, raw = deep_validate(
        key,
        child,
        child_path,
        outer_path,
        outer,
        source,
        verify_checkout=verify_checkout,
    )
    return {
        "outer_path": outer_path,
        "outer": outer,
        "outer_sha256": sha256(outer_path),
        "child_path": child_path,
        "child": child,
        "child_sha256": child_digest,
        "source": source,
        "raw_path": raw,
        "binding": binding,
    }


def cross_bindings(inputs: dict[str, dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    for key in INPUT_SPECS:
        require(inputs[key]["source"] == source, f"{key} source identity differs from S2 source")
    products = {key: inputs[key]["binding"] for key in PRODUCT_KEYS}
    binaries = {binding.get("binary_sha256") for binding in products.values()}
    require(len(binaries) == 1 and None not in binaries, f"S2 product binary identity mismatch: {sorted(str(item) for item in binaries)}")
    revisions = {binding.get("model_revision") for binding in products.values()}
    require(len(revisions) == 1 and None not in revisions, f"S2 model revision mismatch: {sorted(str(item) for item in revisions)}")
    require(all(binding.get("model_id") == MODEL_ID for binding in products.values()), "S2 model identity mismatch")

    hardware_classes = []
    uuids = set()
    drivers = set()
    for key, binding in products.items():
        hardware = require_object(binding.get("hardware"), f"{key} hardware binding")
        hardware_classes.append(
            (hardware.get("device_count"), hardware.get("device_name"), hardware.get("memory_mib"))
        )
        if hardware.get("uuid"):
            uuids.add(hardware["uuid"])
        if hardware.get("driver_version"):
            drivers.add(hardware["driver_version"])
    require(len(set(hardware_classes)) == 1, f"S2 hardware class mismatch: {hardware_classes}")
    require(len(uuids) <= 1, f"S2 artifacts came from different GPU UUIDs: {sorted(uuids)}")
    require(len(drivers) <= 1, f"S2 artifacts used different driver versions: {sorted(drivers)}")

    runner_hashes = {products[key].get("runner_sha256") for key in SCENARIO_KEYS}
    require(len(runner_hashes) == 1 and None not in runner_hashes, "S2 scenario runner identity mismatch")
    validator_refs = {
        key: require_object(inputs[key]["binding"].get("validator"), f"{key} validator")
        for key in INPUT_SPECS
        if isinstance(inputs[key]["binding"], dict) and inputs[key]["binding"].get("validator")
    }
    closure_refs = {
        key: binding["model_file_closure_sha256"]
        for key, binding in products.items()
        if binding.get("model_file_closure_sha256")
    }
    require({"m1_determinism", "latency_first_failure"} <= set(closure_refs), "S2 model file closure anchors are incomplete")
    product_identity = require_object(
        products["multiturn_concurrency"].get("product_execution_identity"),
        "S2 run/serve product execution identity",
    )
    require(
        product_identity.get("entrypoints") == ["run", "serve"]
        and product_identity.get("same_resolved_execution_plan") is True
        and product_identity.get("same_runtime_implementation") is True
        and product_identity.get("production_legacy_selection_count") == 0,
        "S2 run/serve product execution identity failed",
    )
    return {
        "source": source,
        "binary_sha256": next(iter(binaries)),
        "model": {
            "id": MODEL_ID,
            "revision": next(iter(revisions)),
            "file_closure_anchors": closure_refs,
        },
        "hardware": {
            "class": {
                "device_count": hardware_classes[0][0],
                "device_name": hardware_classes[0][1],
                "memory_mib": hardware_classes[0][2],
            },
            "uuid": next(iter(uuids)) if uuids else None,
            "driver_version": next(iter(drivers)) if drivers else None,
            "determinism_hardware_id": products["m1_determinism"].get("hardware_id"),
            "determinism_device_fingerprint": products["m1_determinism"].get("device_fingerprint"),
        },
        "scenario_runner_sha256": next(iter(runner_hashes)),
        "scenario_manifest_sha256_by_lane": {
            key: products[key]["scenario_manifest_sha256"] for key in sorted(SCENARIO_KEYS)
        },
        "effective_config_sha256_by_lane": {
            key: products[key].get("effective_config_sha256", []) for key in sorted(PRODUCT_KEYS)
        },
        "product_execution_identity": copy.deepcopy(product_identity),
        "validators": validator_refs,
    }


def input_reference(item: dict[str, Any], root: Path, key: str) -> dict[str, Any]:
    directory = root / "inputs" / key
    directory.mkdir(parents=True, exist_ok=False)
    outer_copy = directory / "gate.manifest.json"
    child_copy = directory / "manifest.json"
    shutil.copyfile(item["outer_path"], outer_copy)
    shutil.copyfile(item["child_path"], child_copy)
    require(sha256(outer_copy) == item["outer_sha256"], f"{key} outer copy changed")
    require(sha256(child_copy) == item["child_sha256"], f"{key} child copy changed")
    raw = item.get("raw_path")
    return {
        "lane": INPUT_SPECS[key].lane,
        "source": item["source"],
        "outer_manifest": {
            "path": outer_copy.relative_to(root).as_posix(),
            "sha256": item["outer_sha256"],
        },
        "child_manifest": {
            "path": child_copy.relative_to(root).as_posix(),
            "sha256": item["child_sha256"],
        },
        "raw_artifact": {
            "recorded_path": item["child"].get(INPUT_SPECS[key].raw_field)
            if INPUT_SPECS[key].raw_field
            else None,
            "validated_path": str(raw) if isinstance(raw, Path) else None,
        },
        "deep_validation": item["binding"],
    }


def artifact_index(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"S2 aggregate contains a symlink: {path}")
        if not path.is_file() or path == root / "manifest.json":
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def validate_output_layout(manifest: dict[str, Any], root: Path) -> Path:
    recorded_output = Path(
        require_string(manifest.get("output_root"), "S2 output_root")
    )
    recorded_artifact = Path(
        require_string(manifest.get("artifact_dir"), "S2 artifact_dir")
    )
    require(
        recorded_output.is_absolute()
        and recorded_artifact.is_absolute()
        and ".." not in recorded_output.parts
        and ".." not in recorded_artifact.parts,
        "S2 recorded output paths are invalid",
    )
    require(
        recorded_artifact == recorded_output / "s2-product-contract"
        and root.name == "s2-product-contract",
        "S2 aggregate output layout mismatch",
    )
    require(
        manifest.get("pass_line") == f"{PASS_PREFIX}: {recorded_output}",
        "S2 aggregate PASS line mismatch",
    )
    return recorded_output


def verify_checkpoint_manifest(
    manifest_path: Path,
    *,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve()
    manifest = read_json(path, "S2 aggregate manifest")
    root = path.parent
    require(
        set(manifest)
        == {
            "schema_version",
            "artifact_type",
            "checkpoint_id",
            "lane",
            "status",
            "canonical",
            "artifact_dir",
            "output_root",
            "source",
            "dependency_source",
            "source_closure",
            "children",
            "bindings",
            "acceptance",
            "artifact_count",
            "artifact_index_sha256",
            "artifact_index",
            "unlocks",
            "does_not_prove",
            "started_at",
            "finished_at",
            "duration_seconds",
            "pass_line",
        },
        "S2 aggregate manifest field set mismatch",
    )
    require(
        manifest.get("schema_version") == 1
        and manifest.get("artifact_type") == "runtime_vnext_s2_cuda_product_contract_manifest"
        and manifest.get("checkpoint_id") == "S2"
        and manifest.get("lane") == "runtime-vnext-s2"
        and manifest.get("status") == "pass"
        and manifest.get("canonical") is True,
        "S2 aggregate identity/status mismatch",
    )
    validate_output_layout(manifest, root)
    source = require_object(manifest.get("source"), "S2 source")
    if verify_checkout:
        require(source == clean_source(), "S2 aggregate source is stale")
    dependency_source = require_object(
        manifest.get("dependency_source"), "S2 dependency source"
    )
    closure = dependency_source_closure(dependency_source, source)
    require(
        manifest.get("source_closure") == closure,
        "S2 dependency source closure mismatch",
    )
    children = require_object(manifest.get("children"), "S2 children")
    require(set(children) == set(INPUT_SPECS), "S2 child matrix mismatch")
    reconstructed_inputs: dict[str, dict[str, Any]] = {}
    for key, spec in INPUT_SPECS.items():
        ref = require_object(children.get(key), f"S2 child {key}")
        require(
            ref.get("lane") == spec.lane
            and ref.get("source") == dependency_source,
            f"S2 child {key} identity mismatch",
        )
        outer_ref = require_object(ref.get("outer_manifest"), f"S2 child {key} outer")
        child_ref = require_object(ref.get("child_manifest"), f"S2 child {key} child")
        outer_path = safe_relative_path(root, outer_ref["path"], f"S2 child {key} outer")
        child_path = safe_relative_path(root, child_ref["path"], f"S2 child {key} child")
        require(sha256(outer_path) == outer_ref.get("sha256"), f"S2 child {key} outer SHA mismatch")
        require(sha256(child_path) == child_ref.get("sha256"), f"S2 child {key} child SHA mismatch")
        outer = read_json(outer_path, f"S2 copied {key} outer")
        child = read_json(child_path, f"S2 copied {key} child")
        observed_source = validate_outer_child_pair(key, outer, child, child_ref["sha256"])
        require(
            observed_source == dependency_source,
            f"S2 copied {key} source mismatch",
        )
        reconstructed_inputs[key] = {
            "source": dependency_source,
            "binding": ref.get("deep_validation"),
        }
    bindings = cross_bindings(reconstructed_inputs, dependency_source)
    require(manifest.get("bindings") == bindings, "S2 aggregate cross-binding mismatch")
    require(manifest.get("acceptance") == ACCEPTANCE, "S2 acceptance mismatch")
    require(manifest.get("unlocks") == ["S3"], "S2 unlock set mismatch")
    require(manifest.get("does_not_prove") == DOES_NOT_PROVE, "S2 does_not_prove mismatch")
    rows = artifact_index(root)
    require(manifest.get("artifact_count") == len(rows), "S2 artifact count mismatch")
    require(manifest.get("artifact_index") == rows, "S2 artifact index mismatch")
    require(manifest.get("artifact_index_sha256") == canonical_sha256(rows), "S2 artifact index digest mismatch")
    return {
        "kind": "vnext-s2",
        "child_manifest": {
            "path": str(path),
            "sha256": sha256(path),
            "artifact_count": len(rows),
        },
        "source": source,
        "dependency_source": dependency_source,
        "source_closure": closure,
        "bindings": bindings,
    }


def build_checkpoint(input_paths: dict[str, Path], out: Path) -> str:
    source = clean_source()
    output = out.expanduser().resolve()
    require(REPO_ROOT not in output.parents and output != REPO_ROOT, "S2 output must be outside the source tree")
    require(not output.exists() or not any(output.iterdir()), f"S2 output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    root = output / "s2-product-contract"
    root.mkdir(parents=False, exist_ok=False)
    started_at = iso_now()
    started = time.monotonic()
    try:
        inputs = {
            key: load_input(input_paths[key], key, verify_checkout=False)
            for key in INPUT_SPECS
        }
        dependency_source = inputs["g01"]["source"]
        closure = dependency_source_closure(dependency_source, source)
        bindings = cross_bindings(inputs, dependency_source)
        children = {key: input_reference(inputs[key], root, key) for key in INPUT_SPECS}
        rows = artifact_index(root)
        pass_line = f"{PASS_PREFIX}: {output}"
        manifest = {
            "schema_version": 1,
            "artifact_type": "runtime_vnext_s2_cuda_product_contract_manifest",
            "checkpoint_id": "S2",
            "lane": "runtime-vnext-s2",
            "status": "pass",
            "canonical": True,
            "artifact_dir": str(root),
            "output_root": str(output),
            "source": source,
            "dependency_source": dependency_source,
            "source_closure": closure,
            "children": children,
            "bindings": bindings,
            "acceptance": ACCEPTANCE,
            "artifact_count": len(rows),
            "artifact_index_sha256": canonical_sha256(rows),
            "artifact_index": rows,
            "unlocks": ["S3"],
            "does_not_prove": DOES_NOT_PROVE,
            "started_at": started_at,
            "finished_at": iso_now(),
            "duration_seconds": time.monotonic() - started,
            "pass_line": pass_line,
        }
        write_json(root / "manifest.json", manifest)
        verify_checkpoint_manifest(root / "manifest.json", verify_checkout=True)
        return pass_line
    except Exception as error:
        write_json(
            root / "failure.json",
            {
                "schema_version": 1,
                "artifact_type": "runtime_vnext_s2_cuda_product_contract_failure",
                "source": source,
                "started_at": started_at,
                "finished_at": iso_now(),
                "duration_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def expect_reject(action: Callable[[], Any], marker: str) -> None:
    try:
        action()
    except (AggregateError, ValueError) as error:
        require(marker.lower() in str(error).lower(), f"mutation rejected for wrong reason: {error}")
        return
    raise AggregateError(f"mutation unexpectedly passed: {marker}")


def self_test() -> int:
    source_sha = "1" * 40
    source_tree = "2" * 40
    source = {
        "git_sha": source_sha,
        "git_tree_sha": source_tree,
        "dirty": False,
        "status_short": [],
    }
    allowed, rejected = dependency_control_plane_only(
        [
            "scripts/release/runtime_vnext_r0_core_closure.py",
            "scripts/release/runtime_vnext_s2_cuda_product_contract.py",
        ]
    )
    require(len(allowed) == 2 and not rejected, "S2 dependency closure rejected control-plane files")
    _, rejected = dependency_control_plane_only(
        [
            "crates/ferrum-engine/src/lib.rs",
            "scripts/release/scenarios/runtime_vnext_s2_multiturn_concurrency_cuda.json",
        ]
    )
    require(len(rejected) == 2, "S2 dependency closure accepted product or scenario changes")
    require(
        test_evidence_passed({"summary": {"passed": 1}}, "fixture") == 1,
        "nested test evidence count drifted",
    )
    expect_reject(
        lambda: test_evidence_passed({"passed": 1}, "fixture"),
        "summary",
    )
    with tempfile.TemporaryDirectory(prefix="ferrum-vnext-s2-contract-") as temporary:
        root = Path(temporary)
        duplicate = root / "duplicate.json"
        duplicate.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
        expect_reject(lambda: read_json(duplicate, "duplicate fixture"), "duplicate JSON key")
        target = root / "bundle/raw"
        target.mkdir(parents=True)
        outer_path = root / "bundle/gate/gate.manifest.json"
        outer_path.parent.mkdir(parents=True)
        relocated = relocate_recorded_path(
            "/workspace/artifacts/raw",
            outer_path=outer_path,
            recorded_outer_dir="/workspace/artifacts/gate",
            label="relocation fixture",
            directory=True,
        )
        require(relocated == target.resolve(), "relocation fixture failed")
        model_root = root / "model-revision"
        model_root.mkdir()
        model_log = model_root / "server.log"
        model_log.write_text(
            "Path: /cache/hub/models--Qwen--Qwen3.5-4B/snapshots/"
            + "5" * 40
            + "\n",
            encoding="utf-8",
        )
        revision = model_revision_binding("model fixture", model_root, [{"model": MODEL_ID}])
        require(
            revision
            == {
                "model_id": MODEL_ID,
                "model_revision": "5" * 40,
                "resolution_log_sha256": sha256(model_log),
            },
            "model revision log binding failed",
        )
        model_log.write_text(
            model_log.read_text(encoding="utf-8")
            + "Path: /cache/hub/models--Qwen--Qwen3.5-4B/snapshots/"
            + "6" * 40
            + "\n",
            encoding="utf-8",
        )
        expect_reject(
            lambda: model_revision_binding("model fixture", model_root, []),
            "one Qwen3.5-4B revision",
        )
        relocated_root = root / "downloaded/s2-product-contract"
        relocated_root.mkdir(parents=True)
        recorded_output = Path("/workspace/ferrum-artifacts/s2-fixture")
        layout = {
            "output_root": str(recorded_output),
            "artifact_dir": str(recorded_output / "s2-product-contract"),
            "pass_line": f"{PASS_PREFIX}: {recorded_output}",
        }
        require(
            validate_output_layout(layout, relocated_root) == recorded_output,
            "relocated output layout validation failed",
        )
        escaped_layout = copy.deepcopy(layout)
        escaped_layout["artifact_dir"] = "/workspace/ferrum-artifacts/other"
        expect_reject(
            lambda: validate_output_layout(escaped_layout, relocated_root),
            "output layout",
        )

    for key, spec in INPUT_SPECS.items():
        artifact_dir = f"/tmp/{key}"
        child_pass = f"{spec.pass_prefix}: {artifact_dir}"
        child: dict[str, Any] = {
            spec.identity_field: spec.identity_value,
            "status": "pass",
            "pass_line": child_pass,
            "source_git_sha": source_sha,
        }
        outer = {
            "schema_version": 1,
            "lane": spec.lane,
            "status": "pass",
            "git_sha": source_sha,
            "dirty_status": {"is_dirty": False, "status_short": []},
            "artifact_dir": artifact_dir,
            "pass_line": f"FERRUM GATE {spec.lane} PASS: {artifact_dir}",
            "child_pass_line": child_pass,
            "child_artifacts": {
                "kind": spec.child_kind,
                "child_manifest": {"path": f"{artifact_dir}/{spec.child_suffix}", "sha256": "3" * 64},
            },
        }
        original_source_for_sha = globals()["source_for_sha"]
        globals()["source_for_sha"] = lambda _sha: source
        try:
            require(validate_outer_child_pair(key, outer, child, "3" * 64) == source, f"{key} pair validation failed")
            dirty = copy.deepcopy(outer)
            dirty["dirty_status"] = {"is_dirty": True, "status_short": [" M source"]}
            expect_reject(lambda: validate_outer_child_pair(key, dirty, child, "3" * 64), "dirty")
            forged = copy.deepcopy(outer)
            forged["child_artifacts"]["child_manifest"]["sha256"] = "4" * 64
            expect_reject(lambda: validate_outer_child_pair(key, forged, child, "3" * 64), "SHA256")
        finally:
            globals()["source_for_sha"] = original_source_for_sha

    product = {
        "binary_sha256": "4" * 64,
        "model_id": MODEL_ID,
        "model_revision": "5" * 40,
        "hardware": {
            "device_count": 1,
            "device_name": "NVIDIA GeForce RTX 4090",
            "memory_mib": 24564,
            "driver_version": "555.42",
            "uuid": "GPU-fixture",
        },
        "runner_sha256": "6" * 64,
        "scenario_manifest_sha256": "7" * 64,
        "effective_config_sha256": ["8" * 64],
        "validator": {"path": "fixture.py", "sha256": "9" * 64},
    }
    inputs: dict[str, dict[str, Any]] = {}
    for key in INPUT_SPECS:
        binding: dict[str, Any] = {"validator": {"path": f"{key}.py", "sha256": "9" * 64}}
        if key in PRODUCT_KEYS:
            binding = copy.deepcopy(product)
            if key not in SCENARIO_KEYS:
                binding["runner_sha256"] = None
                binding["scenario_manifest_sha256"] = None
            if key in {"m1_determinism", "latency_first_failure"}:
                binding["model_file_closure_sha256"] = ("a" if key == "m1_determinism" else "b") * 64
            if key == "m1_determinism":
                binding["hardware_id"] = "fixture-instance"
                binding["device_fingerprint"] = "c" * 64
            if key == "multiturn_concurrency":
                binding["product_execution_identity"] = {
                    "entrypoints": ["run", "serve"],
                    "resolved_execution_plan_hash": "1" * 64,
                    "runtime_implementation_fingerprint": "2" * 64,
                    "same_resolved_execution_plan": True,
                    "same_runtime_implementation": True,
                    "production_legacy_selection_count": 0,
                }
        inputs[key] = {"source": source, "binding": binding}
    cross_bindings(inputs, source)
    forged = copy.deepcopy(inputs)
    forged["s1_capacity"]["binding"]["binary_sha256"] = "d" * 64
    expect_reject(lambda: cross_bindings(forged, source), "binary")
    forged = copy.deepcopy(inputs)
    forged["tool_schema"]["binding"]["hardware"]["uuid"] = "GPU-other"
    expect_reject(lambda: cross_bindings(forged, source), "UUID")
    forged = copy.deepcopy(inputs)
    forged["response_format"]["binding"]["model_revision"] = "e" * 40
    expect_reject(lambda: cross_bindings(forged, source), "revision")
    forged = copy.deepcopy(inputs)
    forged["multiturn_concurrency"]["binding"]["product_execution_identity"][
        "production_legacy_selection_count"
    ] = 1
    expect_reject(lambda: cross_bindings(forged, source), "product execution identity")
    print(SELFTEST_PASS)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g01", type=Path)
    parser.add_argument("--s1", type=Path)
    parser.add_argument("--s1-capacity", type=Path)
    parser.add_argument("--s1-decode-capacity", type=Path)
    parser.add_argument("--g02-core", type=Path)
    parser.add_argument("--m1-determinism", type=Path)
    parser.add_argument("--response-format", type=Path)
    parser.add_argument("--api-modality", type=Path)
    parser.add_argument("--stream-disconnect", type=Path)
    parser.add_argument("--tool-schema", type=Path)
    parser.add_argument("--multiturn-concurrency", type=Path)
    parser.add_argument("--latency-first-failure", type=Path)
    parser.add_argument("--historical-resource-source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    input_values = {
        "g01": args.g01,
        "s1": args.s1,
        "s1_capacity": args.s1_capacity,
        "s1_decode_capacity": args.s1_decode_capacity,
        "g02_core": args.g02_core,
        "m1_determinism": args.m1_determinism,
        "response_format": args.response_format,
        "api_modality": args.api_modality,
        "stream_disconnect": args.stream_disconnect,
        "tool_schema": args.tool_schema,
        "multiturn_concurrency": args.multiturn_concurrency,
        "latency_first_failure": args.latency_first_failure,
        "historical_resource_source": args.historical_resource_source,
    }
    if args.self_test:
        require(all(value is None for value in input_values.values()) and args.out is None, "--self-test does not accept gate inputs")
    else:
        missing = [key for key, value in input_values.items() if value is None]
        require(not missing and args.out is not None, f"missing required inputs: {', '.join(missing)}")
    args.input_values = input_values
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        if args.self_test:
            return self_test()
        pass_line = build_checkpoint(args.input_values, args.out)
        print(pass_line)
        return 0
    except (AggregateError, OSError, RuntimeError, ValueError) as error:
        print(f"FERRUM RUNTIME VNEXT S2 CUDA PRODUCT CONTRACT FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
