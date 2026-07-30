#!/usr/bin/env python3
"""Build and verify the production-backed G01B reference-contract checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_gate  # noqa: E402
import runtime_vnext_s1_cuda_capacity as capacity_checkpoint  # noqa: E402
import runtime_vnext_s1_cuda_checkpoint as s1_checkpoint  # noqa: E402
import runtime_vnext_s1_cuda_decode_capacity as decode_checkpoint  # noqa: E402


PASS_PREFIX = "FERRUM RUNTIME VNEXT G01B PRODUCTION REFERENCE CONTRACT PASS"
SELFTEST_PASS = "FERRUM RUNTIME VNEXT G01B PRODUCTION REFERENCE CONTRACT SELFTEST PASS"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
BOUNDED_COMMAND = REPO_ROOT / "scripts/release/bounded_command.py"

INPUT_SPECS = {
    "g00f": {
        "lane": "vnext-g00f",
        "artifact_type": "runtime_vnext_g00f_facts_manifest",
    },
    "g01a": {
        "lane": "vnext-g01a",
        "artifact_type": "runtime_vnext_g01a_contract_split_manifest",
    },
    "s1": {
        "lane": "vnext-s1-cuda",
        "artifact_type": "runtime_vnext_s1_cuda_basic_slice_manifest",
    },
    "s1_capacity": {
        "lane": "vnext-s1-cuda-capacity",
        "artifact_type": "runtime_vnext_s1_cuda_capacity_pressure_validation_v2",
    },
    "s1_decode_capacity": {
        "lane": "vnext-s1-cuda-decode-capacity",
        "artifact_type": "runtime_vnext_s1_cuda_decode_capacity_validation",
    },
}

TEST_SPECS = {
    "extension-drills": {
        "target": "vnext_extension_drill_contract_tests",
        "release": False,
        "timeout": 600,
        "expected_tests": {
            "existing_operation_dense_family_is_additive",
            "recurrent_family_reuses_sequence_state_contract",
            "novel_operation_is_additive_to_catalog_provider_and_oracle_graph",
            "reference_backend_reuses_the_same_prepared_model_program",
            "unsupported_backend_fails_with_missing_operation_or_version_before_estimation",
        },
        "proof_lines": set(),
    },
    "source-audit": {
        "target": "vnext_source_audit_contract_tests",
        "release": False,
        "timeout": 600,
        "expected_tests": {
            "generic_contracts_have_zero_architecture_names",
            "silent_success_defaults_are_absent",
            "failure_envelope_wire_limit_precedes_deserialization",
        },
        "proof_lines": set(),
    },
    "plan-snapshots": {
        "target": "vnext_plan_wire_contract_tests",
        "release": False,
        "timeout": 600,
        "expected_tests": {
            "dynamic_descriptor_and_memory_plan_standalone_wire_are_checked",
            "execution_plan_is_deterministic_100_of_100",
            "execution_plan_schema_round_trip_100_of_100",
            "breaking_schema_versions_are_rejected_100_of_100",
            "legacy_schema_is_rejected_before_v8_provider_execution_validation",
            "provider_execution_semantics_are_required_on_the_plan_wire",
            "execution_weight_materializer_and_schema_cannot_authorize_themselves_from_wire",
            "forged_self_hashed_plan_is_rejected_by_semantic_rebuild",
            "resolved_weight_layout_cannot_be_stripped_from_plan_wire",
            "externally_trusted_node_resolution_cannot_be_replaced_by_wire_data",
            "self_consistent_wire_resource_estimate_and_memory_mutation_is_rejected",
            "self_consistent_wire_provider_selection_is_rejected",
            "typed_planning_registry_invokes_real_contract_and_estimator_once",
        },
        "proof_lines": {
            "VNEXT PLAN DETERMINISM PASS: 100/100",
            "VNEXT PLAN ROUNDTRIP PASS: 100/100",
            "VNEXT BREAKING VERSION REJECT PASS: 100/100",
        },
    },
    "overhead": {
        "target": "vnext_g01b_overhead_contract_tests",
        "release": True,
        "timeout": 300,
        "expected_tests": {
            "g01b_preselected_provider_and_event_sink_overhead_are_bounded",
        },
        "proof_lines": {
            "G01B PROVIDER DISPATCH PASS: 30/30",
            "G01B DISABLED EVENT SINK PASS: 30/30",
            "G01B BASIC EVENT SINK PASS: 30/30",
        },
    },
}

PRODUCTION_ROOT_FILES = {
    "Cargo.lock",
    "Cargo.toml",
    "ferrum.toml",
    "rust-toolchain",
    "rust-toolchain.toml",
}
PRODUCTION_EXCLUDED_PARTS = {"tests", "benches", "examples"}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def require_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be a non-empty string")
    return value


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"{label} is invalid JSON: {path}: {error}") from error
    return require_object(value, label)


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
    status = [line for line in git("status", "--short").splitlines() if line.strip()]
    require(not status, f"G01B requires a clean checkout: {status}")
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_tree_sha": git("rev-parse", "HEAD^{tree}"),
        "dirty": False,
        "status_short": [],
    }


def production_path(path: str) -> bool:
    if path in PRODUCTION_ROOT_FILES or path.startswith(".cargo/"):
        return True
    parts = Path(path).parts
    return bool(parts and parts[0] == "crates" and not PRODUCTION_EXCLUDED_PARTS.intersection(parts))


def source_scope(commit: str, predicate: Any) -> dict[str, Any]:
    require(GIT_SHA_RE.fullmatch(commit) is not None, f"invalid source commit: {commit}")
    rows = []
    output = git("ls-tree", "-r", commit, "--", ".cargo", "crates", *sorted(PRODUCTION_ROOT_FILES))
    for line in output.splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, object_id = metadata.split()
        if not predicate(path):
            continue
        rows.append(
            {
                "path": path,
                "mode": mode,
                "kind": kind,
                "git_object": object_id,
            }
        )
    rows.sort(key=lambda row: row["path"])
    require(rows, f"source scope is empty at {commit}")
    return {
        "file_count": len(rows),
        "sha256": canonical_sha256(rows),
        "files": rows,
    }


def production_scope(commit: str) -> dict[str, Any]:
    return source_scope(commit, production_path)


def contract_scope(commit: str) -> dict[str, Any]:
    return source_scope(
        commit,
        lambda path: path.startswith("crates/ferrum-interfaces/src/vnext/")
        and path.endswith(".rs"),
    )


def relocated_path(recorded: str, recorded_root: str, local_root: Path) -> Path:
    recorded_path = Path(recorded)
    root_path = Path(recorded_root)
    try:
        relative = recorded_path.relative_to(root_path)
    except ValueError:
        relative = Path(os.path.relpath(recorded_path, root_path))
    return (local_root / relative).resolve()


def load_gate_input(
    input_path: Path,
    key: str,
) -> dict[str, Any]:
    spec = INPUT_SPECS[key]
    outer_path = input_path.expanduser().resolve()
    outer = read_json(outer_path, f"{key} outer manifest")
    lane = spec["lane"]
    require(outer.get("schema_version") == 1, f"{key} outer schema mismatch")
    require(outer.get("lane") == lane, f"{key} outer lane mismatch")
    require(outer.get("status") == "pass", f"{key} outer status is not pass")
    dirty = require_object(outer.get("dirty_status"), f"{key} outer dirty_status")
    require(dirty.get("is_dirty") is False and dirty.get("status_short") == [], f"{key} outer source was dirty")
    artifact_dir = require_string(outer.get("artifact_dir"), f"{key} outer artifact_dir")
    require(
        outer.get("pass_line") == f"FERRUM GATE {lane} PASS: {artifact_dir}",
        f"{key} outer PASS line mismatch",
    )
    child_artifacts = require_object(outer.get("child_artifacts"), f"{key} child_artifacts")
    child_ref = require_object(child_artifacts.get("child_manifest"), f"{key} child manifest ref")
    child_recorded = require_string(child_ref.get("path"), f"{key} child manifest path")
    child_digest = require_string(child_ref.get("sha256"), f"{key} child manifest SHA256")
    require(SHA256_RE.fullmatch(child_digest) is not None, f"{key} child manifest SHA256 is invalid")
    child_path = relocated_path(child_recorded, artifact_dir, outer_path.parent)
    require(child_path.is_file(), f"{key} relocated child manifest is missing: {child_path}")
    require(sha256(child_path) == child_digest, f"{key} child manifest SHA256 mismatch")
    child = read_json(child_path, f"{key} child manifest")
    require(child.get("artifact_type") == spec["artifact_type"], f"{key} child artifact_type mismatch")
    require(child.get("status") == "pass", f"{key} child status is not pass")
    source_sha = child.get("source_git_sha")
    if source_sha is None:
        source_sha = require_object(child.get("source"), f"{key} child source").get("git_sha")
    require(GIT_SHA_RE.fullmatch(str(source_sha)) is not None, f"{key} child source SHA is invalid")
    require(outer.get("git_sha") == source_sha, f"{key} outer/child source SHA mismatch")
    return {
        "key": key,
        "outer_path": outer_path,
        "outer": outer,
        "outer_sha256": sha256(outer_path),
        "child_path": child_path,
        "child": child,
        "child_sha256": child_digest,
        "source_git_sha": source_sha,
    }


def validate_current_fact_and_contract_inputs(
    g00f: dict[str, Any],
    g01a: dict[str, Any],
    source: dict[str, Any],
) -> None:
    require(g00f["source_git_sha"] == source["git_sha"], "G00F source is stale against current HEAD")
    require(g01a["source_git_sha"] == source["git_sha"], "G01A source is stale against current HEAD")
    g01a_g00f = require_object(g01a["child"].get("g00f"), "G01A G00F binding")
    bound_child = require_object(g01a_g00f.get("child_manifest"), "G01A G00F child binding")
    require(
        bound_child.get("sha256") == g00f["child_sha256"],
        "G01A is not byte-bound to the supplied G00F child manifest",
    )
    require(
        require_object(g01a_g00f.get("source"), "G01A G00F source").get("git_sha")
        == source["git_sha"],
        "G01A G00F source binding is stale",
    )

    g00f_command = run_gate.LaneCommand(
        cmd=[],
        child_manifest_path=g00f["child_path"],
        provenance_kind="vnext-g00f",
    )
    run_gate.validate_vnext_g00f_provenance(
        g00f_command,
        g00f["child"],
        g00f["child_sha256"],
        verify_checkout=True,
    )
    g01a_command = run_gate.LaneCommand(
        cmd=[],
        child_manifest_path=g01a["child_path"],
        provenance_kind="vnext-g01a-s0a",
    )
    run_gate.validate_vnext_g01a_s0a_provenance(
        g01a_command,
        g01a["child"],
        g01a["child_sha256"],
        verify_checkout=True,
    )


def resolve_bound_path(recorded: str, manifest: dict[str, Any], child_path: Path) -> Path:
    artifact_dir = require_string(manifest.get("artifact_dir"), "child artifact_dir")
    return relocated_path(recorded, artifact_dir, child_path.parent)


def model_revision_from_snapshot(path: str) -> str:
    parts = Path(path).parts
    require("snapshots" in parts, f"model path is not a Hugging Face snapshot: {path}")
    index = parts.index("snapshots")
    require(index + 1 < len(parts), f"model snapshot revision is missing: {path}")
    revision = parts[index + 1]
    require(MODEL_REVISION_RE.fullmatch(revision) is not None, f"invalid model snapshot revision: {revision}")
    require(
        "models--Qwen--Qwen3.5-4B" in parts,
        f"model snapshot is not {MODEL_ID}: {path}",
    )
    return revision


def g00f_model_identity(g00f: dict[str, Any]) -> dict[str, Any]:
    g00a_ref = require_object(g00f["child"].get("g00a"), "G00F G00A binding")
    g00a_child_ref = require_object(g00a_ref.get("child_manifest"), "G00F G00A child ref")
    g00a_path = Path(require_string(g00a_child_ref.get("path"), "G00F G00A child path")).resolve()
    require(g00a_path.is_file(), f"G00F bound G00A manifest is unavailable: {g00a_path}")
    require(
        sha256(g00a_path) == require_string(g00a_child_ref.get("sha256"), "G00F G00A child SHA256"),
        "G00F bound G00A manifest SHA256 mismatch",
    )
    g00a = read_json(g00a_path, "G00F bound G00A manifest")
    index = {
        row["path"]: row
        for row in g00a.get("artifact_index", [])
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    resolution_ref = require_object(index.get("model-resolution.json"), "G00A model-resolution index")
    resolution_path = g00a_path.parent / "model-resolution.json"
    require(
        sha256(resolution_path) == resolution_ref.get("sha256"),
        "G00A model-resolution SHA256 mismatch",
    )
    resolution = read_json(resolution_path, "G00A model resolution")
    matches = [
        row
        for row in resolution.get("lanes", [])
        if isinstance(row, dict)
        and row.get("backend") == "cuda"
        and require_object(row.get("semantic_source"), "model semantic source").get("repo") == MODEL_ID
    ]
    require(len(matches) == 1, f"G00A must contain exactly one CUDA {MODEL_ID} lane")
    lane = matches[0]
    semantic = require_object(lane.get("semantic_source"), "G00A model semantic source")
    weight = require_object(lane.get("weight_source"), "G00A model weight source")
    revision = require_string(semantic.get("revision"), "G00A model semantic revision")
    require(MODEL_REVISION_RE.fullmatch(revision) is not None, "G00A model revision is invalid")
    require(weight.get("repo") == MODEL_ID and weight.get("revision") == revision, "G00A CUDA weight/semantic source mismatch")
    return {
        "model_id": MODEL_ID,
        "revision": revision,
        "catalog_lane_id": lane.get("catalog_lane_id"),
        "format": lane.get("format"),
        "model_resolution": {
            "path": str(resolution_path),
            "sha256": sha256(resolution_path),
        },
    }


def normalized_manifest(value: dict[str, Any], ignored: set[str]) -> dict[str, Any]:
    return {key: child for key, child in value.items() if key not in ignored}


def semantic_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            semantic_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            semantic_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    return left == right


def revalidate_basic(item: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    child = item["child"]
    require(child.get("model_id") == MODEL_ID, "S1 basic model id mismatch")
    require(child.get("backend") == "cuda", "S1 basic backend mismatch")
    require(set(child.get("entrypoints", [])) == {"ferrum run", "ferrum serve"}, "S1 basic entrypoint matrix mismatch")
    raw = resolve_bound_path(
        require_string(child.get("raw_artifact_dir"), "S1 basic raw_artifact_dir"),
        child,
        item["child_path"],
    )
    require(raw.is_dir(), f"S1 basic raw artifact is missing: {raw}")
    collection = read_json(raw / "collection.json", "S1 basic collection")
    revision = model_revision_from_snapshot(
        require_string(collection.get("model_snapshot_path"), "S1 basic model snapshot")
    )
    require(revision == model["revision"], "S1 basic model revision differs from G00F")
    summary = s1_checkpoint.validate(raw, item["source_git_sha"])
    s1_checkpoint.require_bounded_overhead_native_evidence(summary)
    performance = s1_checkpoint.validate_profile_overhead(raw)
    validation_ref = require_object(child.get("validation"), "S1 basic validation ref")
    validation_path = item["child_path"].parent / require_string(
        validation_ref.get("path"),
        "S1 basic validation path",
    )
    require(
        sha256(validation_path) == validation_ref.get("sha256"),
        "S1 basic validation SHA256 mismatch",
    )
    saved_validation = read_json(validation_path, "S1 basic validation")
    with tempfile.TemporaryDirectory(prefix="ferrum-g01b-basic-") as temporary:
        regenerated_root = Path(temporary)
        s1_checkpoint.write_basic_slice_evidence(raw, regenerated_root, summary, performance)
        regenerated = read_json(regenerated_root / "manifest.json", "regenerated S1 basic manifest")
        regenerated_validation = read_json(
            regenerated_root / "validation.json",
            "regenerated S1 basic validation",
        )
    require(
        semantic_equal(
            normalized_manifest(regenerated_validation, {"raw_artifact_dir"}),
            normalized_manifest(saved_validation, {"raw_artifact_dir"}),
        ),
        "S1 basic validation differs from raw evidence revalidation",
    )
    ignored = {"artifact_dir", "raw_artifact_dir", "pass_line", "validation"}
    require(
        semantic_equal(
            normalized_manifest(regenerated, ignored),
            normalized_manifest(child, ignored),
        ),
        "S1 basic child manifest differs from raw evidence revalidation",
    )
    return {
        "source_git_sha": item["source_git_sha"],
        "binary_sha256": child.get("binary_sha256"),
        "model_revision": revision,
        "run_correctness": True,
        "serve_correctness": True,
        "stream_correctness": True,
        "bench_serve_correctness": True,
        "mean_profile_overhead_fraction": child["metrics"]["mean_overhead_fraction"],
        "median_profile_overhead_fraction": child["metrics"]["median_overhead_fraction"],
        "raw_artifact_dir": str(raw),
        "raw_artifact_index_sha256": saved_validation["raw_artifact_index_sha256"],
    }


def revalidate_capacity(
    item: dict[str, Any],
    model: dict[str, Any],
    *,
    decode: bool,
) -> dict[str, Any]:
    child = item["child"]
    recorded_raw = Path(
        require_string(child.get("source_artifact"), f"{item['key']} source_artifact")
    )
    candidates = [
        recorded_raw,
        item["child_path"].parent.parent / "raw",
        item["child_path"].parent / "raw",
    ]
    raw = next((candidate.resolve() for candidate in candidates if candidate.is_dir()), candidates[1].resolve())
    require(raw.is_dir(), f"{item['key']} raw artifact is missing: {raw}")
    revision = model_revision_from_snapshot(require_string(child.get("model_path"), f"{item['key']} model_path"))
    require(revision == model["revision"], f"{item['key']} model revision differs from G00F")
    with tempfile.TemporaryDirectory(prefix=f"ferrum-g01b-{item['key']}-") as temporary:
        output = Path(temporary)
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            result = (
                decode_checkpoint.validate(raw, output)
                if decode
                else capacity_checkpoint.validate(raw, output)
            )
        require(result == 0, f"{item['key']} raw revalidation returned {result}")
        regenerated = read_json(output / "manifest.json", f"regenerated {item['key']} manifest")
    ignored = {"source_artifact", "pass_line"}
    require(
        semantic_equal(
            normalized_manifest(regenerated, ignored),
            normalized_manifest(child, ignored),
        ),
        f"{item['key']} child manifest differs from raw evidence revalidation",
    )
    return {
        "source_git_sha": item["source_git_sha"],
        "binary_sha256": child.get("binary_sha256"),
        "model_revision": revision,
        "raw_artifact_dir": str(raw.resolve()),
        "source_collection_sha256": child.get("source_collection_sha256"),
        "dynamic_admission_backpressure": True,
        "active_decode_progress_under_pressure": True,
        "release_epoch_resume": True,
        "decode_capacity": decode,
    }


def validate_cuda_inputs(
    inputs: dict[str, dict[str, Any]],
    current_production: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    scopes = {}
    for key in ("s1", "s1_capacity", "s1_decode_capacity"):
        item = inputs[key]
        scope = production_scope(item["source_git_sha"])
        require(
            scope["sha256"] == current_production["sha256"],
            f"{key} production source is stale against current HEAD",
        )
        scopes[key] = {
            "source_git_sha": item["source_git_sha"],
            "file_count": scope["file_count"],
            "sha256": scope["sha256"],
        }
    return {
        "production_source_scope": {
            "current": {
                "file_count": current_production["file_count"],
                "sha256": current_production["sha256"],
            },
            "inputs": scopes,
        },
        "basic": revalidate_basic(inputs["s1"], model),
        "capacity": revalidate_capacity(inputs["s1_capacity"], model, decode=False),
        "decode_capacity": revalidate_capacity(
            inputs["s1_decode_capacity"],
            model,
            decode=True,
        ),
    }


def validate_bounded_receipt(path: Path) -> dict[str, Any]:
    receipt = read_json(path, "bounded command receipt")
    require(
        receipt.get("schema") == "ferrum.bounded-command-receipt.v1"
        and receipt.get("status") == "pass"
        and receipt.get("rc") == 0
        and receipt.get("reason") == "command_completed"
        and receipt.get("violation") is None
        and receipt.get("sampling_errors") == []
        and receipt.get("termination") == {"signals": [], "errors": []}
        and receipt.get("cleanup") == {"process_group_gone": True},
        f"bounded command did not complete cleanly: {path}",
    )
    return receipt


def run_contract_test(checkpoint_root: Path, name: str) -> dict[str, Any]:
    spec = TEST_SPECS[name]
    logs = checkpoint_root / "logs" / name
    logs.mkdir(parents=True, exist_ok=False)
    receipt_path = logs / "receipt.json"
    stdout_path = logs / "stdout.log"
    stderr_path = logs / "stderr.log"
    command = [
        "cargo",
        "test",
        "-p",
        "ferrum-interfaces",
    ]
    if spec["release"]:
        command.append("--release")
    command.extend(
        [
            "--test",
            spec["target"],
            "--",
            "--test-threads=1",
            "--nocapture",
        ]
    )
    bounded = [
        sys.executable,
        str(BOUNDED_COMMAND),
        "--receipt",
        str(receipt_path),
        "--stdout-log",
        str(stdout_path),
        "--stderr-log",
        str(stderr_path),
        "--cwd",
        str(REPO_ROOT),
        "--wall-timeout-seconds",
        str(spec["timeout"]),
        "--max-processes",
        "32",
        "--max-group-threads",
        "96",
        "--max-per-process-threads",
        "32",
        "--sample-interval-seconds",
        "0.1",
        "--max-sampling-errors",
        "3",
        "--term-grace-seconds",
        "2",
        "--",
        *command,
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CARGO_BUILD_JOBS": "4",
            "RUST_TEST_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    started_at = iso_now()
    started = time.monotonic()
    result = subprocess.run(
        bounded,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    duration = time.monotonic() - started
    (logs / "runner.stdout").write_text(result.stdout, encoding="utf-8")
    (logs / "runner.stderr").write_text(result.stderr, encoding="utf-8")
    require(result.returncode == 0, f"{name} bounded runner failed with rc={result.returncode}")
    receipt = validate_bounded_receipt(receipt_path)
    output = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    combined = f"{output}\n{stderr}"
    require("test result: FAILED" not in combined, f"{name} contains a failed test result")
    observed_tests = set(
        re.findall(r"^test ([A-Za-z0-9_]+) \.\.\. ok$", combined, flags=re.MULTILINE)
    )
    require(observed_tests == spec["expected_tests"], f"{name} exact test set mismatch: {sorted(observed_tests)}")
    summary_pattern = (
        rf"test result: ok\. {len(spec['expected_tests'])} passed; "
        r"0 failed; 0 ignored; 0 measured; 0 filtered out;"
    )
    require(len(re.findall(summary_pattern, combined)) == 1, f"{name} exact test summary mismatch")
    lines = [line.strip() for line in combined.splitlines()]
    for proof in spec["proof_lines"]:
        require(lines.count(proof) == 1, f"{name} missing or duplicate proof line: {proof}")
    return {
        "schema_version": 1,
        "target": spec["target"],
        "release": spec["release"],
        "command": command,
        "env_overrides": {
            "CARGO_BUILD_JOBS": "4",
            "RUST_TEST_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "started_at": started_at,
        "finished_at": iso_now(),
        "duration_seconds": duration,
        "tests": sorted(observed_tests),
        "test_count": len(observed_tests),
        "proof_lines": sorted(spec["proof_lines"]),
        "receipt": {
            "path": receipt_path.relative_to(checkpoint_root).as_posix(),
            "sha256": sha256(receipt_path),
            "limits": receipt["limits"],
            "peaks": receipt["peaks"],
            "cleanup": receipt["cleanup"],
        },
        "stdout": {
            "path": stdout_path.relative_to(checkpoint_root).as_posix(),
            "sha256": sha256(stdout_path),
            "size_bytes": stdout_path.stat().st_size,
        },
        "stderr": {
            "path": stderr_path.relative_to(checkpoint_root).as_posix(),
            "sha256": sha256(stderr_path),
            "size_bytes": stderr_path.stat().st_size,
        },
        "status": "pass",
        "_combined": combined,
    }


def parse_overhead(evidence: dict[str, Any]) -> dict[str, Any]:
    matches = re.findall(r"^G01B OVERHEAD JSON: (\{.*\})$", evidence.pop("_combined"), re.MULTILINE)
    require(len(matches) == 1, "overhead command must emit exactly one JSON report")
    report = require_object(json.loads(matches[0]), "G01B overhead report")
    require(report.get("sample_count") == 30, "G01B overhead sample count mismatch")
    disabled = report.get("disabled_sink_median_overhead")
    basic = report.get("basic_sink_median_overhead")
    provider = report.get("provider_median_overhead")
    delta = report.get("provider_median_delta_per_call_ns")
    require(isinstance(disabled, (int, float)) and disabled <= 0.01, "disabled sink overhead exceeds 1%")
    require(isinstance(basic, (int, float)) and basic <= 0.02, "basic sink overhead exceeds 2%")
    require(
        isinstance(provider, (int, float))
        and isinstance(delta, (int, float))
        and (provider <= 0.01 or delta <= 2.0),
        "provider dispatch overhead exceeds both accepted bounds",
    )
    return {
        "schema_version": 1,
        "artifact_type": "runtime_vnext_g01b_overhead_evidence",
        "status": "pass",
        "command_evidence": evidence,
        "report": report,
        "acceptance": {
            "disabled_sink_median_overhead_lte_1pct": True,
            "basic_sink_median_overhead_lte_2pct": True,
            "provider_dispatch_negligible": True,
        },
    }


def copy_input_manifests(
    checkpoint_root: Path,
    inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    references = {}
    for key, item in inputs.items():
        directory = (
            checkpoint_root / "run-serve-evidence" / key
            if key.startswith("s1")
            else checkpoint_root / "inputs" / key
        )
        directory.mkdir(parents=True, exist_ok=False)
        outer_copy = directory / "gate.manifest.json"
        child_copy = directory / "manifest.json"
        shutil.copyfile(item["outer_path"], outer_copy)
        shutil.copyfile(item["child_path"], child_copy)
        require(sha256(outer_copy) == item["outer_sha256"], f"{key} copied outer manifest changed")
        require(sha256(child_copy) == item["child_sha256"], f"{key} copied child manifest changed")
        references[key] = {
            "lane": INPUT_SPECS[key]["lane"],
            "source_git_sha": item["source_git_sha"],
            "outer_manifest": {
                "path": outer_copy.relative_to(checkpoint_root).as_posix(),
                "sha256": item["outer_sha256"],
            },
            "child_manifest": {
                "path": child_copy.relative_to(checkpoint_root).as_posix(),
                "sha256": item["child_sha256"],
            },
        }
    return references


def artifact_index(checkpoint_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(checkpoint_root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" and path.parent == checkpoint_root:
            continue
        rows.append(
            {
                "path": path.relative_to(checkpoint_root).as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def build_gate(
    *,
    g00f_path: Path,
    g01a_path: Path,
    s1_path: Path,
    s1_capacity_path: Path,
    s1_decode_capacity_path: Path,
    output_root: Path,
) -> str:
    source = clean_source()
    output = output_root.expanduser().resolve()
    require(REPO_ROOT not in output.parents and output != REPO_ROOT, "G01B output must be outside the source tree")
    checkpoint_root = output / "g01b-reference-contract"
    require(not checkpoint_root.exists(), f"G01B checkpoint output already exists: {checkpoint_root}")
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    started_at = iso_now()
    started = time.monotonic()
    try:
        inputs = {
            "g00f": load_gate_input(g00f_path, "g00f"),
            "g01a": load_gate_input(g01a_path, "g01a"),
            "s1": load_gate_input(s1_path, "s1"),
            "s1_capacity": load_gate_input(s1_capacity_path, "s1_capacity"),
            "s1_decode_capacity": load_gate_input(
                s1_decode_capacity_path,
                "s1_decode_capacity",
            ),
        }
        validate_current_fact_and_contract_inputs(inputs["g00f"], inputs["g01a"], source)
        current_production = production_scope(source["git_sha"])
        current_contract = contract_scope(source["git_sha"])
        model = g00f_model_identity(inputs["g00f"])
        cuda = validate_cuda_inputs(inputs, current_production, model)
        copied_inputs = copy_input_manifests(checkpoint_root, inputs)

        extension = run_contract_test(checkpoint_root, "extension-drills")
        extension.pop("_combined")
        source_audit = run_contract_test(checkpoint_root, "source-audit")
        source_audit.pop("_combined")
        plans = run_contract_test(checkpoint_root, "plan-snapshots")
        plans.pop("_combined")
        overhead = parse_overhead(run_contract_test(checkpoint_root, "overhead"))

        extension_document = {
            "schema_version": 1,
            "artifact_type": "runtime_vnext_g01b_extension_drills",
            "status": "pass",
            "drills_passed": 5,
            "drills_required": 5,
            "shared_runtime_changes_required": 0,
            "evidence": extension,
        }
        write_json(checkpoint_root / "extension-drills.json", extension_document)
        plan_document = {
            "schema_version": 1,
            "artifact_type": "runtime_vnext_g01b_plan_snapshots",
            "status": "pass",
            "deterministic": {"passed": 100, "required": 100},
            "schema_round_trip": {"passed": 100, "required": 100},
            "breaking_version_rejected": {"passed": 100, "required": 100},
            "source_audit": source_audit,
            "evidence": plans,
        }
        write_json(checkpoint_root / "plan-snapshots" / "summary.json", plan_document)
        write_json(checkpoint_root / "overhead.json", overhead)
        product_document = {
            "schema_version": 1,
            "artifact_type": "runtime_vnext_g01b_qwen35_4b_cuda_production",
            "status": "pass",
            "model": model,
            "cuda": cuda,
            "legacy_fallback_count": 0,
            "entrypoints": ["ferrum run", "ferrum serve"],
            "dynamic_capacity_evidence": ["s1_capacity", "s1_decode_capacity"],
        }
        write_json(checkpoint_root / "qwen35-4b-cuda-production.json", product_document)

        pass_line = f"{PASS_PREFIX}: {output}"
        rows = artifact_index(checkpoint_root)
        manifest = {
            "schema_version": 1,
            "artifact_type": "runtime_vnext_g01b_production_reference_contract_manifest",
            "checkpoint_id": "G01B-S0B-S1",
            "lane": "runtime-vnext-g01b",
            "status": "pass",
            "canonical": True,
            "artifact_dir": str(checkpoint_root),
            "output_root": str(output),
            "source": source,
            "production_source_scope": {
                "file_count": current_production["file_count"],
                "sha256": current_production["sha256"],
            },
            "contract_source_scope": {
                "file_count": current_contract["file_count"],
                "sha256": current_contract["sha256"],
            },
            "inputs": copied_inputs,
            "model": model,
            "evidence": {
                "product": {
                    "path": "qwen35-4b-cuda-production.json",
                    "sha256": sha256(checkpoint_root / "qwen35-4b-cuda-production.json"),
                },
                "extension_drills": {
                    "path": "extension-drills.json",
                    "sha256": sha256(checkpoint_root / "extension-drills.json"),
                },
                "plan_snapshots": {
                    "path": "plan-snapshots/summary.json",
                    "sha256": sha256(checkpoint_root / "plan-snapshots" / "summary.json"),
                },
                "overhead": {
                    "path": "overhead.json",
                    "sha256": sha256(checkpoint_root / "overhead.json"),
                },
            },
            "acceptance": {
                "g00f_current_and_byte_bound": True,
                "g01a_current_and_byte_bound": True,
                "same_contract_source": True,
                "qwen35_4b_cuda_run": True,
                "qwen35_4b_cuda_serve": True,
                "stream_and_bench_serve_correctness": True,
                "dynamic_admission_backpressure": True,
                "decode_continues_under_prefill_pressure": True,
                "release_epoch_resume": True,
                "legacy_fallback_count_zero": True,
                "extension_drills_5_of_5": True,
                "architecture_names_in_generic_contract_zero": True,
                "silent_success_defaults_zero": True,
                "plan_determinism_100_of_100": True,
                "plan_round_trip_100_of_100": True,
                "breaking_version_rejection_100_of_100": True,
                "disabled_event_sink_overhead_lte_1pct": True,
                "basic_event_sink_overhead_lte_2pct": True,
                "provider_dispatch_overhead_negligible": True,
            },
            "artifact_count": len(rows),
            "artifact_index_sha256": canonical_sha256(rows),
            "artifact_index": rows,
            "unlocks": ["G01"],
            "does_not_prove": ["G01", "S1", "full_model_migration", "release"],
            "started_at": started_at,
            "finished_at": iso_now(),
            "duration_seconds": time.monotonic() - started,
            "pass_line": pass_line,
        }
        write_json(checkpoint_root / "manifest.json", manifest)
        verify_checkpoint_manifest(checkpoint_root / "manifest.json", verify_checkout=True)
        return pass_line
    except Exception as error:
        write_json(
            checkpoint_root / "failure.json",
            {
                "schema_version": 1,
                "artifact_type": "runtime_vnext_g01b_failure",
                "source": source,
                "started_at": started_at,
                "finished_at": iso_now(),
                "duration_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def verify_checkpoint_manifest(
    manifest_path: Path,
    *,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve()
    manifest = read_json(path, "G01B manifest")
    root = path.parent
    require(manifest.get("schema_version") == 1, "G01B schema mismatch")
    require(
        manifest.get("artifact_type") == "runtime_vnext_g01b_production_reference_contract_manifest",
        "G01B artifact_type mismatch",
    )
    require(
        manifest.get("checkpoint_id") == "G01B-S0B-S1"
        and manifest.get("lane") == "runtime-vnext-g01b"
        and manifest.get("status") == "pass"
        and manifest.get("canonical") is True,
        "G01B checkpoint identity/status mismatch",
    )
    require(Path(require_string(manifest.get("artifact_dir"), "G01B artifact_dir")).resolve() == root, "G01B artifact_dir mismatch")
    output = Path(require_string(manifest.get("output_root"), "G01B output_root")).resolve()
    require(root == output / "g01b-reference-contract", "G01B output/checkpoint layout mismatch")
    require(manifest.get("pass_line") == f"{PASS_PREFIX}: {output}", "G01B PASS line mismatch")
    source = require_object(manifest.get("source"), "G01B source")
    require(
        source.get("dirty") is False and source.get("status_short") == [],
        "G01B source dirty state mismatch",
    )
    source_sha = require_string(source.get("git_sha"), "G01B source git SHA")
    require(GIT_SHA_RE.fullmatch(source_sha) is not None, "G01B source SHA is invalid")
    if verify_checkout:
        current = clean_source()
        require(current["git_sha"] == source_sha, "G01B source SHA is stale against current HEAD")
        require(current["git_tree_sha"] == source.get("git_tree_sha"), "G01B source tree is stale")

    current_production = production_scope(source_sha)
    current_contract = contract_scope(source_sha)
    require(
        manifest.get("production_source_scope")
        == {
            "file_count": current_production["file_count"],
            "sha256": current_production["sha256"],
        },
        "G01B production source scope mismatch",
    )
    require(
        manifest.get("contract_source_scope")
        == {
            "file_count": current_contract["file_count"],
            "sha256": current_contract["sha256"],
        },
        "G01B contract source scope mismatch",
    )

    inputs = require_object(manifest.get("inputs"), "G01B inputs")
    require(set(inputs) == set(INPUT_SPECS), "G01B input matrix mismatch")
    for key, spec in INPUT_SPECS.items():
        ref = require_object(inputs.get(key), f"G01B input {key}")
        require(ref.get("lane") == spec["lane"], f"G01B input {key} lane mismatch")
        for kind in ("outer_manifest", "child_manifest"):
            file_ref = require_object(ref.get(kind), f"G01B input {key} {kind}")
            relative = require_string(file_ref.get("path"), f"G01B input {key} {kind} path")
            file_path = (root / relative).resolve()
            require(root in file_path.parents, f"G01B input path escapes checkpoint: {relative}")
            require(sha256(file_path) == file_ref.get("sha256"), f"G01B input {key} {kind} SHA mismatch")
        copied_child = read_json(root / ref["child_manifest"]["path"], f"G01B copied {key} child")
        require(copied_child.get("artifact_type") == spec["artifact_type"], f"G01B copied {key} type mismatch")

    evidence = require_object(manifest.get("evidence"), "G01B evidence")
    expected_evidence = {
        "product": "qwen35-4b-cuda-production.json",
        "extension_drills": "extension-drills.json",
        "plan_snapshots": "plan-snapshots/summary.json",
        "overhead": "overhead.json",
    }
    require(set(evidence) == set(expected_evidence), "G01B evidence matrix mismatch")
    documents = {}
    for key, relative in expected_evidence.items():
        ref = require_object(evidence.get(key), f"G01B {key} evidence ref")
        require(ref.get("path") == relative, f"G01B {key} evidence path mismatch")
        evidence_path = root / relative
        require(sha256(evidence_path) == ref.get("sha256"), f"G01B {key} evidence SHA mismatch")
        documents[key] = read_json(evidence_path, f"G01B {key} evidence")
        require(documents[key].get("status") == "pass", f"G01B {key} evidence is not pass")

    product = documents["product"]
    require(
        product.get("legacy_fallback_count") == 0
        and set(product.get("entrypoints", [])) == {"ferrum run", "ferrum serve"},
        "G01B product entrypoint/fallback evidence mismatch",
    )
    model = require_object(product.get("model"), "G01B product model")
    require(
        model.get("model_id") == MODEL_ID
        and MODEL_REVISION_RE.fullmatch(str(model.get("revision"))) is not None,
        "G01B product model identity mismatch",
    )
    require(manifest.get("model") == model, "G01B manifest/product model identity mismatch")
    extension = documents["extension_drills"]
    require(
        extension.get("drills_passed") == 5
        and extension.get("drills_required") == 5
        and extension.get("shared_runtime_changes_required") == 0,
        "G01B extension drill evidence mismatch",
    )
    plans = documents["plan_snapshots"]
    for key in ("deterministic", "schema_round_trip", "breaking_version_rejected"):
        require(plans.get(key) == {"passed": 100, "required": 100}, f"G01B plan {key} evidence mismatch")
    overhead = documents["overhead"]
    require(
        overhead.get("acceptance")
        == {
            "disabled_sink_median_overhead_lte_1pct": True,
            "basic_sink_median_overhead_lte_2pct": True,
            "provider_dispatch_negligible": True,
        },
        "G01B overhead acceptance mismatch",
    )
    acceptance = require_object(manifest.get("acceptance"), "G01B acceptance")
    require(acceptance and all(value is True for value in acceptance.values()), "G01B acceptance contains a non-true value")
    require(manifest.get("unlocks") == ["G01"], "G01B unlock matrix mismatch")
    require("G01" in manifest.get("does_not_prove", []), "G01B must not independently prove G01")

    rows = artifact_index(root)
    require(manifest.get("artifact_count") == len(rows), "G01B artifact count mismatch")
    require(manifest.get("artifact_index") == rows, "G01B artifact index mismatch")
    require(manifest.get("artifact_index_sha256") == canonical_sha256(rows), "G01B artifact index digest mismatch")
    return {
        "kind": "vnext-g01b",
        "child_manifest": {
            "path": str(path),
            "sha256": sha256(path),
            "artifact_count": len(rows),
        },
        "source": {
            "git_sha": source_sha,
            "git_tree_sha": source.get("git_tree_sha"),
        },
        "production_source_scope": manifest["production_source_scope"],
        "contract_source_scope": manifest["contract_source_scope"],
        "model": manifest["model"],
        "inputs": {
            key: {
                "source_git_sha": value["source_git_sha"],
                "child_manifest_sha256": value["child_manifest"]["sha256"],
            }
            for key, value in inputs.items()
        },
    }


def self_test() -> int:
    require(set(INPUT_SPECS) == {"g00f", "g01a", "s1", "s1_capacity", "s1_decode_capacity"}, "G01B input matrix drifted")
    require(set(TEST_SPECS) == {"extension-drills", "source-audit", "plan-snapshots", "overhead"}, "G01B test matrix drifted")
    require(len(TEST_SPECS["extension-drills"]["expected_tests"]) == 5, "G01B extension drill count drifted")
    require(len(TEST_SPECS["source-audit"]["expected_tests"]) == 3, "G01B source audit count drifted")
    require(len(TEST_SPECS["plan-snapshots"]["expected_tests"]) == 13, "G01B plan test count drifted")
    require(TEST_SPECS["overhead"]["release"] is True, "G01B overhead must use release mode")
    require(production_path("crates/ferrum-engine/src/lib.rs"), "production source classification rejected crate src")
    require(production_path("crates/ferrum-kernels/cuda/kernel.cu"), "production source classification rejected native source")
    require(production_path("Cargo.lock"), "production source classification rejected Cargo.lock")
    for excluded in (
        "scripts/release/run_gate.py",
        "docs/goals/goal.md",
        "crates/ferrum-engine/tests/regression.rs",
        "crates/ferrum-engine/benches/bench.rs",
        "crates/ferrum-engine/examples/example.rs",
    ):
        require(not production_path(excluded), f"production source classification included {excluded}")
    with tempfile.TemporaryDirectory(prefix="ferrum-g01b-selftest-") as temporary:
        local_root = Path(temporary) / "gate"
        local_root.mkdir()
        expected = (local_root / "../raw").resolve()
        actual = relocated_path(
            "/workspace/evidence/raw",
            "/workspace/evidence/gate",
            local_root,
        )
        require(actual == expected, "G01B relocated sibling path policy drifted")
    require(
        model_revision_from_snapshot(
            "/workspace/hf-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/"
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        )
        == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "G01B model revision parser drifted",
    )
    print(SELFTEST_PASS)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g00f", type=Path)
    parser.add_argument("--g01a", type=Path)
    parser.add_argument("--s1", type=Path)
    parser.add_argument("--s1-capacity", type=Path)
    parser.add_argument("--s1-decode-capacity", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        try:
            return self_test()
        except (GateError, OSError, ValueError) as error:
            print(f"{SELFTEST_PASS} FAIL: {error}", file=sys.stderr)
            return 1
    required = {
        "--g00f": args.g00f,
        "--g01a": args.g01a,
        "--s1": args.s1,
        "--s1-capacity": args.s1_capacity,
        "--s1-decode-capacity": args.s1_decode_capacity,
        "--out": args.out,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("required arguments: " + ", ".join(missing))
    try:
        print(
            build_gate(
                g00f_path=args.g00f,
                g01a_path=args.g01a,
                s1_path=args.s1,
                s1_capacity_path=args.s1_capacity,
                s1_decode_capacity_path=args.s1_decode_capacity,
                output_root=args.out,
            )
        )
        return 0
    except (
        GateError,
        OSError,
        ValueError,
        run_gate.GateError,
        s1_checkpoint.ValidationError,
        capacity_checkpoint.CapacityGateError,
    ) as error:
        print(f"{PASS_PREFIX} FAIL: {args.out}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
