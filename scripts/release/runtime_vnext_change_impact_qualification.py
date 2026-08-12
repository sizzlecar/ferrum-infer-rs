#!/usr/bin/env python3
"""Qualify a focused post-R1 change without replaying unaffected R1 cells."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable

import plan_gates


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1
ARTIFACT_TYPE = "runtime_vnext_change_impact_qualification_manifest"
WITNESS_TYPE = "runtime_vnext_change_impact_witness"
RAW_EVIDENCE_TYPE = "runtime_vnext_change_impact_raw_execution"
PROFILE_ID = "vnext-profile-timing-observability"
LEGACY_PROFILE_ID = "vnext-reusable-startup-diagnostic-observability"
PASS_PREFIX = "FERRUM RUNTIME VNEXT CHANGE IMPACT QUALIFICATION PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT CHANGE IMPACT QUALIFICATION SELFTEST PASS"
CONTROL_SELFTEST_PASS_LINE = (
    "FERRUM RUNTIME VNEXT CHANGE IMPACT CONTROL SELFTEST PASS"
)
EXACT_CONTRACTS_PASS_LINE = (
    "FERRUM RUNTIME VNEXT PROFILE TIMING EXACT CONTRACTS PASS"
)
PROFILE_COLLECTOR_SELFTEST_PASS_LINE = (
    "FERRUM RUNTIME VNEXT PROFILE COLLECTOR FOCUSED SELFTEST PASS"
)
LEGACY_UNIT_TEST_PASS_LINE = (
    "FERRUM RUNTIME VNEXT CHANGE IMPACT EXACT UNIT PASS"
)
LEGACY_EXACT_UNIT_FILTER = (
    "executor::vnext_executor::tests::"
    "reusable_program_identity_is_recorded_before_catalog_installation"
)
PLANNER_SELFTEST_PASS_LINE = "CHANGE IMPACT GATE PLAN SELFTEST PASS"
R1_SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT R1 PRODUCT CORRECTNESS SELFTEST PASS"
HOST_SUSPEND_SELFTEST_PASS_LINE = (
    "FERRUM RUNTIME VNEXT HOST SUSPEND ASSEMBLER SELFTEST PASS"
)
RULES_PATH = REPO_ROOT / "scripts/release/change_impact_rules.json"
FIXTURES_PATH = (
    REPO_ROOT / "scripts/release/fixtures/change_impact/planner_fixtures.json"
)
PRODUCT_SCENARIO_PATH = (
    REPO_ROOT / "scripts/release/scenarios/product_regression.json"
)
LEGACY_EXECUTOR_PATH = "crates/ferrum-models/src/executor/vnext_executor.rs"
SMOKE_SCENARIO_PATH = (
    REPO_ROOT / "scripts/release/scenarios/product_regression_smoke.json"
)
PLANNER_PATH = REPO_ROOT / "scripts/release/plan_gates.py"
CANONICAL_INPUT_PATHS = {
    "qualification": "scripts/release/runtime_vnext_change_impact_qualification.py",
    "rules": "scripts/release/change_impact_rules.json",
    "planner": "scripts/release/plan_gates.py",
    "planner_fixtures": "scripts/release/fixtures/change_impact/planner_fixtures.json",
    "product_regression": "scripts/release/scenarios/product_regression.json",
    "product_regression_smoke": "scripts/release/scenarios/product_regression_smoke.json",
    "r1_validator": "scripts/release/runtime_vnext_r1_product_correctness.py",
    "host_suspend_validator": "scripts/release/runtime_vnext_baseline_scenarios.py",
    "unified_gate": "scripts/release/run_gate.py",
}
EXPECTED_CHECKS = [
    "cuda_run_profile_full",
    "profile_collector_selftest",
    "profile_timing_exact_contracts",
]
EXPECTED_SCOPES = [
    {"backend": "cuda", "entrypoint": "run", "profile_detail": "full"},
]
EXPECTED_CONTROL_GATES = [
    "docs_review",
    "planner_selftest",
    "release_validator_selftest",
]
LEGACY_EXPECTED_CONTROL_GATES = [
    "docs_review",
    "planner_selftest",
    "release_validator_selftest",
]
LEGACY_EXPECTED_CHECKS = [
    "cuda_run_profile_full",
    "cuda_serve_profile_verify",
    "executor_unit_reusable_program_identity",
]
LEGACY_EXPECTED_SCOPES = [
    {"backend": "cuda", "entrypoint": "run", "profile_detail": "full"},
    {"backend": "cuda", "entrypoint": "serve", "profile_detail": "verify"},
]
QUALIFICATION_CONTRACTS = {
    (SCHEMA_VERSION, ARTIFACT_TYPE, PROFILE_ID): "profile-timing-v1",
    (SCHEMA_VERSION, ARTIFACT_TYPE, LEGACY_PROFILE_ID): "legacy-startup-v1",
}
QUALIFICATION_MANIFEST_FIELDS = {
    "schema_version",
    "artifact_type",
    "status",
    "profile_id",
    "artifact_dir",
    "source",
    "prior_source",
    "prior_r1",
    "canonical_inputs",
    "diff",
    "classification",
    "proofs",
    "reused_cells",
    "revalidated_cells",
    "invalidated_cells",
    "open_invalidated_cells",
    "backend_binary_sha256",
    "prior_reachability",
    "does_not_prove",
    "created_at",
    "pass_line",
}
PROFILE_TIMING_MANIFEST_FIELDS = QUALIFICATION_MANIFEST_FIELDS | {
    "witness_hardware"
}
CONTROL_PLANE_RULE_ID = "change-impact-control-plane"
CONTROL_PLANE_PATHS = (
    "scripts/release/change_impact_rules.json",
    "scripts/release/fixtures/change_impact/planner_fixtures.json",
    "scripts/release/plan_gates.py",
    "scripts/release/runtime_vnext_change_impact_qualification.py",
    "scripts/release/runtime_vnext_r1_product_correctness.py",
    "scripts/release/runtime_vnext_r2_performance_build_profile.py",
)
CONTROL_PLANE_DOMAINS = ["change_impact_control_plane"]
CONTROL_PLANE_REQUIRED_GATES = [
    "planner_selftest",
    "release_validator_selftest",
]
MATRIX_CELL_ORDER = (
    "m1_cuda",
    "m1_metal",
    "m2_cuda",
    "m2_metal",
    "m3_cuda",
    "m3_metal",
)
OLD_STARTUP_ERROR = (
    "reusable execution catalog contains a physical program not observed"
)
MIN_STAGE_COVERAGE = 0.90
MAX_CLOCK_RELATIVE_ERROR = 0.005
MIN_DISPATCH_TIMING_COVERAGE = 0.80
MIN_DEVICE_ATTRIBUTION_COVERAGE = 0.80
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DOES_NOT_PROVE = [
    "R2 profile stage coverage, performance, or build-time acceptance",
    "R3 exact staged-binary acceptance",
    "v0.8.0 release readiness",
]


class QualificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"invalid {label} JSON {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    require(
        resolved.is_file() and not resolved.is_symlink(),
        f"artifact is missing: {resolved}",
    )
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def validate_ref(value: Any, label: str) -> Path:
    require(
        isinstance(value, dict)
        and set(value) == {"path", "sha256", "size_bytes"},
        f"{label} reference fields differ",
    )
    path = Path(str(value["path"])).expanduser().resolve()
    require(path.is_file() and not path.is_symlink(), f"{label} is missing: {path}")
    require(path.stat().st_size == value["size_bytes"], f"{label} size mismatch")
    require(sha256(path) == value["sha256"], f"{label} SHA256 mismatch")
    return path


def require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} is invalid",
    )
    return value


def git_process(args: list[str], *, text: bool = False) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=text,
        check=False,
    )


def git_bytes(*args: str) -> bytes:
    process = git_process(list(args))
    require(
        process.returncode == 0,
        f"git {' '.join(args)} failed: {process.stderr.decode('utf-8', errors='replace').strip()}",
    )
    return bytes(process.stdout)


def git_text(*args: str) -> str:
    process = git_process(list(args), text=True)
    require(
        process.returncode == 0,
        f"git {' '.join(args)} failed: {process.stderr.strip()}",
    )
    return process.stdout.strip()


def normalize_source(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} source is missing")
    source = {
        "git_sha": value.get("git_sha"),
        "git_tree_sha": value.get("git_tree_sha"),
        "dirty": value.get("dirty"),
    }
    require(
        isinstance(source["git_sha"], str)
        and GIT_SHA_RE.fullmatch(source["git_sha"]) is not None
        and isinstance(source["git_tree_sha"], str)
        and GIT_SHA_RE.fullmatch(source["git_tree_sha"]) is not None
        and source["dirty"] is False,
        f"{label} source identity is invalid",
    )
    require(
        git_text("rev-parse", f"{source['git_sha']}^{{tree}}")
        == source["git_tree_sha"],
        f"{label} source tree differs from Git",
    )
    return source


def source_at(sha: str) -> dict[str, Any]:
    require(GIT_SHA_RE.fullmatch(sha) is not None, "Git source SHA is invalid")
    require(git_text("cat-file", "-t", sha) == "commit", "Git source is not a commit")
    return {
        "git_sha": sha,
        "git_tree_sha": git_text("rev-parse", f"{sha}^{{tree}}"),
        "dirty": False,
    }


def current_source() -> dict[str, Any]:
    dirty = [line for line in git_text("status", "--short").splitlines() if line]
    require(not dirty, f"qualification source must be clean: {dirty[:8]}")
    return source_at(git_text("rev-parse", "HEAD"))


def control_selftest_command(source: dict[str, Any]) -> list[str]:
    return [
        "python3",
        "-B",
        "scripts/release/runtime_vnext_change_impact_qualification.py",
        "--control-self-test",
        "--expected-source-sha",
        source["git_sha"],
    ]


def profile_timing_exact_contracts_command(source: dict[str, Any]) -> list[str]:
    return [
        "python3",
        "-B",
        "scripts/release/runtime_vnext_change_impact_qualification.py",
        "--run-profile-timing-exact-contracts",
        "--expected-source-sha",
        source["git_sha"],
    ]


def profile_collector_selftest_command(source: dict[str, Any]) -> list[str]:
    return [
        "python3",
        "-B",
        "scripts/release/runtime_vnext_change_impact_qualification.py",
        "--run-profile-collector-selftest",
        "--expected-source-sha",
        source["git_sha"],
    ]


def legacy_exact_unit_test_command(source: dict[str, Any]) -> list[str]:
    return [
        "python3",
        "-B",
        "scripts/release/runtime_vnext_change_impact_qualification.py",
        "--run-exact-unit-test",
        "--expected-source-sha",
        source["git_sha"],
    ]


def changed_paths(base_sha: str, head_sha: str) -> list[str]:
    raw = git_bytes("diff", "--name-only", "-z", f"{base_sha}..{head_sha}", "--")
    return sorted(item.decode("utf-8") for item in raw.split(b"\0") if item)


def git_file_bytes(sha: str, path: str) -> bytes:
    return git_bytes("show", f"{sha}:{path}")


def git_blob_ref(source: dict[str, Any], repo_path: str) -> dict[str, Any]:
    content = git_file_bytes(source["git_sha"], repo_path)
    blob_sha = git_text("rev-parse", f"{source['git_sha']}:{repo_path}")
    require(GIT_SHA_RE.fullmatch(blob_sha) is not None, f"Git blob SHA is invalid: {repo_path}")
    return {
        "repo_path": repo_path,
        "git_blob_sha": blob_sha,
        "sha256": sha256_bytes(content),
        "size_bytes": len(content),
    }


def canonical_input_refs(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        label: git_blob_ref(source, repo_path)
        for label, repo_path in CANONICAL_INPUT_PATHS.items()
    }


def validate_canonical_input_refs(
    value: Any, source: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    expected = canonical_input_refs(source)
    require(value == expected, "qualification source-bound canonical inputs differ")
    return expected


def historical_planner(source: dict[str, Any]) -> tuple[tempfile.TemporaryDirectory[str], ModuleType, Path]:
    """Load planner semantics and rule inputs from the qualified source commit."""

    temporary = tempfile.TemporaryDirectory(prefix="ferrum-qualified-planner-")
    root = Path(temporary.name)
    paths = {
        repo_path: root / repo_path for repo_path in CANONICAL_INPUT_PATHS.values()
    }
    for repo_path, output in paths.items():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(git_file_bytes(source["git_sha"], repo_path))
    module_path = paths[CANONICAL_INPUT_PATHS["planner"]]
    module_name = f"ferrum_qualified_plan_gates_{source['git_sha']}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    require(spec is not None and spec.loader is not None, "cannot load qualified planner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return temporary, module, paths[CANONICAL_INPUT_PATHS["rules"]]


def is_release_product_path(path: str) -> bool:
    return (
        path.startswith("crates/")
        or path.startswith("native-operators/")
        or path.startswith(".cargo/")
        or path in {"Cargo.toml", "Cargo.lock", "ferrum.toml"}
        or path.startswith("rust-toolchain")
    )


def release_product_projection(
    build_source: dict[str, Any], qualified_source: dict[str, Any]
) -> dict[str, Any]:
    changed = changed_paths(build_source["git_sha"], qualified_source["git_sha"])
    product_changes = [path for path in changed if is_release_product_path(path)]
    ignored_test_changes: list[str] = []
    rejected: list[str] = []
    for path in product_changes:
        if not path.endswith(".rs"):
            rejected.append(path)
            continue
        try:
            before = git_file_bytes(build_source["git_sha"], path).decode("utf-8")
            after = git_file_bytes(qualified_source["git_sha"], path).decode("utf-8")
        except UnicodeDecodeError as error:
            raise QualificationError(f"product source is not UTF-8: {path}") from error
        before_product, before_error = plan_gates.production_text(before, True)
        after_product, after_error = plan_gates.production_text(after, True)
        require(
            before_product is not None and after_product is not None,
            f"Rust test boundary is unsafe for {path}: "
            f"build={before_error}, qualification={after_error}",
        )
        if before_product == after_product:
            ignored_test_changes.append(path)
        else:
            rejected.append(path)
    require(
        not rejected,
        f"candidate build source differs in release product inputs: {rejected}",
    )
    return {
        "from_source": build_source,
        "to_source": qualified_source,
        "changed_paths": changed,
        "release_product_changed_paths": product_changes,
        "test_only_paths": ignored_test_changes,
        "equivalent": True,
    }


def recursive_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from recursive_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from recursive_strings(item)


def derive_reused_cells(acceptance: dict[str, Any]) -> list[dict[str, Any]]:
    models = acceptance.get("models")
    require(isinstance(models, dict), "prior R1 model acceptance is missing")
    cells: list[dict[str, Any]] = []
    total = 0
    for key in MATRIX_CELL_ORDER:
        row = models.get(key)
        require(isinstance(row, dict), f"prior R1 {key} acceptance is missing")
        cases = row.get("cases")
        require(isinstance(cases, str) and "/" in cases, f"prior R1 {key} cases differ")
        passed, denominator = (int(item) for item in cases.split("/", 1))
        require(passed == denominator and denominator > 0, f"prior R1 {key} did not pass")
        total += denominator
        cells.append(
            {
                "cell_id": f"r1.matrix.{key}",
                "backend": row.get("backend"),
                "mode": "profile_off",
                "evidence": "correctness",
                "case_count": denominator,
            }
        )
    require(
        total == acceptance.get("total_matrix_case_count") == 1867,
        "prior R1 matrix denominator differs",
    )
    for backend in ("cuda", "metal"):
        cells.append(
            {
                "cell_id": f"r1.llama.{backend}",
                "backend": backend,
                "mode": "profile_off",
                "evidence": "correctness",
                "scenario_count": 3,
            }
        )
    return cells


def full_r1_anchor(
    path: Path, manifest: dict[str, Any], seen: set[Path] | None = None
) -> tuple[Path, dict[str, Any]]:
    current_path = path.expanduser().resolve()
    current_manifest = manifest
    visited = set() if seen is None else set(seen)
    for _depth in range(129):
        require(
            current_path not in visited,
            "prior R1 cumulative dependency cycle detected",
        )
        visited.add(current_path)
        dependencies = current_manifest.get("dependencies")
        require(isinstance(dependencies, dict), "prior R1 dependencies are missing")
        keys = set(dependencies)
        if keys == {"r0", "matrices", "llama_dense", "acceptance"}:
            return current_path, current_manifest
        require(
            keys == {"prior_r1", "impact_qualification", "acceptance"},
            "prior R1 dependency shape is unsupported",
        )
        current_path = validate_ref(
            dependencies.get("prior_r1"), "prior cumulative R1"
        )
        current_manifest = read_json(current_path, "prior cumulative R1")
    raise QualificationError("prior R1 cumulative depth exceeds 128")


def prior_r1_summary(
    path: Path, *, _verification_context: Any | None = None
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    manifest = read_json(resolved, "prior R1 manifest")
    require(
        manifest.get("artifact_type") == "runtime_vnext_r1_product_correctness_manifest"
        and manifest.get("checkpoint_id") == "R1"
        and manifest.get("status") == "pass"
        and manifest.get("canonical") is True,
        "prior R1 is not a canonical PASS",
    )
    source = normalize_source(manifest.get("source"), "prior R1")
    try:
        import runtime_vnext_r1_product_correctness as r1_correctness

        context = r1_correctness._coerce_verification_context(
            _verification_context
        )

        verified = r1_correctness.verify_manifest(
            resolved,
            verify_checkout=False,
            expected_source=source,
            _verification_context=context,
        )
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise QualificationError(f"prior R1 deep verification failed: {error}") from error
    require(
        isinstance(verified, dict) and verified.get("kind") == "vnext-r1",
        "prior R1 deep verifier returned an invalid summary",
    )
    acceptance = verified.get("acceptance")
    require(isinstance(acceptance, dict), "prior R1 acceptance is missing")
    require(
        acceptance.get("matrix_lanes") == "6/6"
        and acceptance.get("matrix_failure_count") == 0
        and acceptance.get("llama_dense_backends") == "2/2"
        and acceptance.get("llama_dense_scenarios") == "6/6"
        and acceptance.get("waiver_count") == 0
        and acceptance.get("exception_count") == 0,
        "prior R1 acceptance denominator/status differs",
    )
    binaries = acceptance.get("backend_binary_sha256")
    require(
        isinstance(binaries, dict) and set(binaries) == {"cuda", "metal"},
        "prior R1 backend binary authority is missing",
    )
    for backend in ("cuda", "metal"):
        require_sha256(binaries[backend], f"prior R1 {backend} binary")
    r1_key = r1_correctness._artifact_key("r1", resolved)
    r1_meta = context.r1_meta.get(r1_key)
    require(isinstance(r1_meta, dict), "prior R1 verification metadata is missing")
    full_path = Path(str(r1_meta.get("full_anchor_path", ""))).resolve()
    full_key = r1_meta.get("full_anchor_key")
    require(
        isinstance(full_key, tuple)
        and full_key in context.r1_memo
        and full_path.is_file(),
        "prior R1 full-anchor metadata is invalid",
    )
    full_manifest = read_json(full_path, "full R1 anchor")
    full_acceptance = full_manifest.get("acceptance")
    require(isinstance(full_acceptance, dict), "full R1 anchor acceptance is missing")
    evidence_binaries = acceptance.get(
        "evidence_backend_binary_sha256", binaries
    )
    require(
        evidence_binaries == full_acceptance.get("backend_binary_sha256")
        and acceptance.get("backend_hardware_id")
        == full_acceptance.get("backend_hardware_id"),
        "prior R1 cumulative chain drifted from its full evidence anchor",
    )
    reused_cells = derive_reused_cells(acceptance)
    reachability = context.reachability_memo.get(full_key)
    if reachability is None:
        profile_flag_count = 0
        hidden_profile_env_count = 0
        full_dependencies = full_manifest.get("dependencies")
        require(isinstance(full_dependencies, dict), "full R1 anchor dependencies are missing")
        matrices = full_dependencies.get("matrices")
        llamas = full_dependencies.get("llama_dense")
        require(isinstance(matrices, dict) and isinstance(llamas, dict), "prior R1 evidence maps are missing")
        evidence_refs: list[dict[str, Any]] = []
        for key in MATRIX_CELL_ORDER:
            row = matrices.get(key)
            require(isinstance(row, dict), f"prior R1 {key} evidence is missing")
            evidence_refs.append(row.get("scenario_report"))
        for backend in ("cuda", "metal"):
            row = llamas.get(backend)
            require(isinstance(row, dict), f"prior R1 Llama {backend} evidence is missing")
            evidence_refs.append(row.get("execution_receipt"))
        for index, ref in enumerate(evidence_refs):
            evidence_path = validate_ref(ref, f"prior R1 reachability evidence[{index}]")
            evidence = read_json(evidence_path, f"prior R1 reachability evidence[{index}]")
            strings = list(recursive_strings(evidence))
            profile_flag_count += sum(item == "--profile-detail" for item in strings)
            hidden_profile_env_count += sum(
                item.startswith("FERRUM_") and "PROFILE" in item for item in strings
            )
        require(
            profile_flag_count == 0 and hidden_profile_env_count == 0,
            "prior R1 evidence unexpectedly enabled profile-only behavior",
        )
        reachability = {
            "matrix_case_count": 1867,
            "llama_scenario_count": 6,
            "profile_flag_count": profile_flag_count,
            "hidden_profile_env_count": hidden_profile_env_count,
            "mode": "profile_off",
        }
        context.reachability_memo[full_key] = copy.deepcopy(reachability)
    return {
        "source": source,
        "acceptance": copy.deepcopy(acceptance),
        "backend_binary_sha256": copy.deepcopy(binaries),
        "evidence_backend_binary_sha256": copy.deepcopy(evidence_binaries),
        "full_r1_anchor": file_ref(full_path),
        "reused_cells": reused_cells,
        "reachability": copy.deepcopy(reachability),
    }


def validate_candidate_build(
    binary_path: Path,
    receipt_path: Path,
    qualified_source: dict[str, Any],
    *,
    require_native_op_artifact: bool = False,
) -> dict[str, Any]:
    binary = file_ref(binary_path)
    receipt = read_json(receipt_path, "candidate CUDA build receipt")
    require(
        receipt.get("schema_version") == 1
        and receipt.get("artifact_type") == "runtime_vnext_candidate_build_receipt"
        and receipt.get("status") == "pass"
        and receipt.get("backend") == "cuda"
        and receipt.get("returncode") == 0,
        "candidate CUDA build receipt identity/status differs",
    )
    build_source = normalize_source(
        {
            "git_sha": receipt.get("source_git_sha"),
            "git_tree_sha": receipt.get("source_tree_sha"),
            "dirty": False,
        },
        "candidate CUDA build",
    )
    require(
        receipt.get("dirty_status") == {"is_dirty": False, "status_short": []},
        "candidate CUDA build was dirty",
    )
    binary_sha = require_sha256(
        receipt.get("binary_sha256"), "candidate CUDA build binary"
    )
    require(binary_sha == binary["sha256"], "candidate CUDA binary differs from build receipt")
    command = receipt.get("command")
    require(isinstance(command, list) and all(isinstance(item, str) for item in command), "candidate build command is invalid")
    command_text = " ".join(command)
    for marker in (
        "cargo build",
        "--release",
        "--features",
        "cuda",
        "vllm-moe-marlin",
        "vllm-paged-attn-v2",
    ):
        require(marker in command_text, f"candidate CUDA build command lacks {marker}")
    if require_native_op_artifact:
        bounded_path = validate_ref(
            receipt.get("bounded_receipt"),
            "candidate CUDA bounded build receipt",
        )
        bounded = read_json(bounded_path, "candidate CUDA bounded build receipt")
        require(
            bounded.get("schema") == "ferrum.bounded-command-receipt.v1"
            and bounded.get("status") == "pass"
            and bounded.get("rc") == 0
            and bounded.get("reason") == "command_completed"
            and bounded.get("command") == command
            and bounded.get("violation") is None
            and bounded.get("cleanup") == {"process_group_gone": True}
            and bounded.get("termination") == {"errors": [], "signals": []}
            and bounded.get("sampling_error_count") == 0
            and bounded.get("sampling_errors") == [],
            "candidate CUDA bounded build receipt did not prove the recorded command",
        )
        for marker in ("--locked", "native-op-artifact"):
            require(
                marker in command_text,
                f"candidate CUDA build command lacks {marker}",
            )
        native_lock_path = validate_ref(
            receipt.get("native_operator_set_lock"),
            "candidate CUDA native operator set lock",
        )
        try:
            import runtime_vnext_g07a_build_iteration as build_iteration

            native_closure = build_iteration.native_operator_set_closure(
                native_lock_path
            )
        except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise QualificationError(
                f"candidate CUDA native operator closure failed: {error}"
            ) from error
        require(native_closure, "candidate CUDA native operator closure is empty")
        native_operator_set_lock = file_ref(native_lock_path)
        native_operator_closure_sha256 = sha256_bytes(
            json.dumps(
                native_closure,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
        bounded_receipt = file_ref(bounded_path)
    else:
        bounded_receipt = None
        native_operator_set_lock = None
        native_operator_closure_sha256 = None
    projection = release_product_projection(build_source, qualified_source)
    result = {
        "binary": binary,
        "build_receipt": file_ref(receipt_path),
        "build_source": build_source,
        "release_product_projection": projection,
        "binary_sha256": binary_sha,
    }
    if require_native_op_artifact:
        result["bounded_receipt"] = bounded_receipt
        result["native_operator_set_lock"] = native_operator_set_lock
        result["native_operator_closure_sha256"] = (
            native_operator_closure_sha256
        )
    return result


def validate_unit_receipt(
    path: Path, qualified_source: dict[str, Any]
) -> dict[str, Any]:
    receipt = read_json(path, "exact unit receipt")
    require(
        receipt.get("schema") == "ferrum.bounded-command-receipt.v1"
        and receipt.get("status") == "pass"
        and receipt.get("rc") == 0
        and receipt.get("reason") == "command_completed",
        "exact unit receipt did not pass",
    )
    require(
        receipt.get("command") == legacy_exact_unit_test_command(qualified_source),
        "exact unit receipt command/source binding differs",
    )
    cwd = Path(str(receipt.get("cwd", ""))).expanduser().resolve()
    require(
        cwd == REPO_ROOT,
        "exact unit test ran outside the repository",
    )
    require(
        receipt.get("violation") is None
        and receipt.get("cleanup") == {"process_group_gone": True}
        and receipt.get("termination") == {"errors": [], "signals": []}
        and receipt.get("sampling_error_count") == 0
        and receipt.get("sampling_errors") == [],
        "exact unit test was not cleanly contained",
    )
    stdout_path = validate_ref(receipt.get("stdout"), "exact unit stdout")
    stderr_path = validate_ref(receipt.get("stderr"), "exact unit stderr")
    stdout_lines = stdout_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    require(
        stdout_lines.count(LEGACY_UNIT_TEST_PASS_LINE) == 1
        and stdout_lines[-1:] == [LEGACY_UNIT_TEST_PASS_LINE],
        "exact unit source-bound PASS line is missing",
    )
    return {
        "check_id": "executor_unit_reusable_program_identity",
        "receipt": file_ref(path),
        "stdout": file_ref(stdout_path),
        "stderr": file_ref(stderr_path),
        "pass_line": LEGACY_UNIT_TEST_PASS_LINE,
        "source": copy.deepcopy(qualified_source),
        "test_source": git_blob_ref(qualified_source, LEGACY_EXECUTOR_PATH),
    }


def validate_source_bound_bounded_receipt(
    path: Path,
    qualified_source: dict[str, Any],
    *,
    check_id: str,
    expected_command: list[str],
    pass_line: str,
) -> dict[str, Any]:
    receipt = read_json(path, f"{check_id} receipt")
    require(
        receipt.get("schema") == "ferrum.bounded-command-receipt.v1"
        and receipt.get("status") == "pass"
        and receipt.get("rc") == 0
        and receipt.get("reason") == "command_completed"
        and receipt.get("command") == expected_command,
        f"{check_id} receipt identity/status differs",
    )
    require(
        Path(str(receipt.get("cwd", ""))).expanduser().resolve() == REPO_ROOT,
        f"{check_id} ran outside the repository",
    )
    require(
        receipt.get("violation") is None
        and receipt.get("cleanup") == {"process_group_gone": True}
        and receipt.get("termination") == {"errors": [], "signals": []}
        and receipt.get("sampling_error_count") == 0
        and receipt.get("sampling_errors") == [],
        f"{check_id} was not cleanly contained",
    )
    stdout_path = validate_ref(receipt.get("stdout"), f"{check_id} stdout")
    stderr_path = validate_ref(receipt.get("stderr"), f"{check_id} stderr")
    stdout_lines = stdout_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    require(
        stdout_lines.count(pass_line) == 1 and stdout_lines[-1:] == [pass_line],
        f"{check_id} exact PASS line is missing",
    )
    return {
        "check_id": check_id,
        "receipt": file_ref(path),
        "stdout": file_ref(stdout_path),
        "stderr": file_ref(stderr_path),
        "pass_line": pass_line,
        "source": copy.deepcopy(qualified_source),
    }


def validate_profile_timing_exact_contracts_receipt(
    path: Path, qualified_source: dict[str, Any]
) -> dict[str, Any]:
    return validate_source_bound_bounded_receipt(
        path,
        qualified_source,
        check_id="profile_timing_exact_contracts",
        expected_command=profile_timing_exact_contracts_command(qualified_source),
        pass_line=EXACT_CONTRACTS_PASS_LINE,
    )


def validate_profile_collector_selftest_receipt(
    path: Path, qualified_source: dict[str, Any]
) -> dict[str, Any]:
    return validate_source_bound_bounded_receipt(
        path,
        qualified_source,
        check_id="profile_collector_selftest",
        expected_command=profile_collector_selftest_command(qualified_source),
        pass_line=PROFILE_COLLECTOR_SELFTEST_PASS_LINE,
    )


def validate_control_gate_receipt(
    path: Path, qualified_source: dict[str, Any]
) -> dict[str, Any]:
    receipt = read_json(path, "qualification control self-test receipt")
    require(
        receipt.get("schema") == "ferrum.bounded-command-receipt.v1"
        and receipt.get("status") == "pass"
        and receipt.get("rc") == 0
        and receipt.get("reason") == "command_completed"
        and receipt.get("command") == control_selftest_command(qualified_source),
        "qualification control self-test receipt identity/status differs",
    )
    cwd = Path(str(receipt.get("cwd", ""))).expanduser().resolve()
    require(cwd == REPO_ROOT, "qualification control self-test ran outside the repository")
    require(
        receipt.get("violation") is None
        and receipt.get("cleanup") == {"process_group_gone": True}
        and receipt.get("termination") == {"errors": [], "signals": []}
        and receipt.get("sampling_error_count") == 0
        and receipt.get("sampling_errors") == [],
        "qualification control self-test was not cleanly contained",
    )
    stdout_path = validate_ref(
        receipt.get("stdout"), "qualification control self-test stdout"
    )
    stderr_path = validate_ref(
        receipt.get("stderr"), "qualification control self-test stderr"
    )
    stdout_lines = stdout_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    require(
        stdout_lines.count(PLANNER_SELFTEST_PASS_LINE) == 1
        and stdout_lines.count(SELFTEST_PASS_LINE) == 1
        and stdout_lines.count(R1_SELFTEST_PASS_LINE) == 1
        and stdout_lines.count(HOST_SUSPEND_SELFTEST_PASS_LINE) == 1
        and stdout_lines.count(CONTROL_SELFTEST_PASS_LINE) == 1
        and stdout_lines[-1:] == [CONTROL_SELFTEST_PASS_LINE],
        "qualification control self-test exact PASS line is missing",
    )
    return {
        "gates": ["planner_selftest", "release_validator_selftest"],
        "source": copy.deepcopy(qualified_source),
        "receipt": file_ref(path),
        "stdout": file_ref(stdout_path),
        "stderr": file_ref(stderr_path),
        "pass_line": CONTROL_SELFTEST_PASS_LINE,
    }


def direct_product_command(
    argv: Any,
    *,
    check_id: str,
    entrypoint: str,
    profile_detail: str,
) -> list[str]:
    require(
        isinstance(argv, list)
        and len(argv) >= 2
        and all(isinstance(item, str) and item for item in argv),
        f"{check_id} raw command argv is invalid",
    )
    require(
        Path(argv[0]).name == "ferrum" and argv[1] == entrypoint,
        f"{check_id} raw command is not direct ferrum {entrypoint}",
    )
    profile_values: list[str] = []
    index = 2
    while index < len(argv):
        item = argv[index]
        if item == "--profile-detail":
            require(
                index + 1 < len(argv),
                f"{check_id} raw command profile detail has no value",
            )
            profile_values.append(argv[index + 1])
            index += 2
            continue
        if item.startswith("--profile-detail="):
            profile_values.append(item.split("=", 1)[1])
        index += 1
    require(
        profile_values == [profile_detail],
        f"{check_id} raw command profile detail differs: {profile_values}",
    )
    return copy.deepcopy(argv)


def strict_process_returncode(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} return code is invalid",
    )
    return value


def flag_values(argv: list[str], flag: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == flag:
            require(index + 1 < len(argv), f"{flag} has no value")
            values.append(argv[index + 1])
            index += 2
            continue
        if item.startswith(f"{flag}="):
            values.append(item.split("=", 1)[1])
        index += 1
    return values


def exact_flag_value(argv: list[str], flag: str, label: str) -> str:
    values = flag_values(argv, flag)
    require(len(values) == 1 and values[0], f"{label} must contain one {flag}")
    return values[0]


def validate_copied_log(
    receipt_value: Any, copied_value: Any, label: str
) -> Path:
    copied_path = validate_ref(copied_value, f"{label} copy")
    require(
        isinstance(receipt_value, dict)
        and receipt_value.get("sha256") == copied_value.get("sha256")
        and receipt_value.get("size_bytes") == copied_value.get("size_bytes"),
        f"{label} copy differs from bounded receipt",
    )
    return copied_path


def parse_utc(value: Any, label: str) -> datetime:
    require(isinstance(value, str) and value, f"{label} timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise QualificationError(f"{label} timestamp is invalid: {value}") from error
    require(parsed.tzinfo is not None, f"{label} timestamp lacks timezone")
    return parsed


def unwrap_sanitized_command(command: Any, label: str) -> list[str]:
    require(
        isinstance(command, list)
        and len(command) >= 4
        and all(isinstance(item, str) and item for item in command),
        f"{label} command is invalid",
    )
    require(
        Path(command[0]).name == "env" and command[1] == "-i",
        f"{label} did not run through an empty environment",
    )
    index = 2
    environment: dict[str, str] = {}
    while index < len(command) and "=" in command[index]:
        key, value = command[index].split("=", 1)
        require(
            key and key not in environment and not key.startswith("FERRUM_"),
            f"{label} sanitized environment is invalid",
        )
        environment[key] = value
        index += 1
    require(index < len(command), f"{label} has no product command")
    return copy.deepcopy(command[index:])


def validate_bounded_execution_receipt(
    receipt_ref: Any,
    stdout_ref: Any,
    stderr_ref: Any,
    label: str,
) -> dict[str, Any]:
    receipt_path = validate_ref(receipt_ref, f"{label} bounded receipt")
    receipt = read_json(receipt_path, f"{label} bounded receipt")
    limits = receipt.get("limits")
    require(
        receipt.get("schema") == "ferrum.bounded-command-receipt.v1"
        and receipt.get("status") == "pass"
        and receipt.get("reason") == "command_completed"
        and receipt.get("rc") == 0
        and receipt.get("violation") is None
        and receipt.get("cleanup") == {"process_group_gone": True}
        and receipt.get("termination") == {"errors": [], "signals": []}
        and receipt.get("sampling_error_count") == 0
        and receipt.get("sampling_errors") == []
        and isinstance(limits, dict)
        and isinstance(limits.get("wall_timeout_seconds"), (int, float))
        and limits["wall_timeout_seconds"] > 0
        and isinstance(limits.get("max_processes"), int)
        and 0 < limits["max_processes"] <= 64
        and isinstance(limits.get("max_group_threads"), int)
        and 0 < limits["max_group_threads"] <= 1024
        and isinstance(limits.get("max_per_process_threads"), int)
        and 0 < limits["max_per_process_threads"] <= 512,
        f"{label} bounded execution did not pass cleanly",
    )
    stdout_path = validate_copied_log(
        receipt.get("stdout"), stdout_ref, f"{label} stdout"
    )
    stderr_path = validate_copied_log(
        receipt.get("stderr"), stderr_ref, f"{label} stderr"
    )
    started = parse_utc(receipt.get("started_at"), f"{label} start")
    ended = parse_utc(receipt.get("ended_at"), f"{label} end")
    require(started <= ended, f"{label} bounded time range is invalid")
    return {
        "receipt": file_ref(receipt_path),
        "receipt_body": receipt,
        "command": unwrap_sanitized_command(receipt.get("command"), label),
        "stdout": stdout_path,
        "stderr": stderr_path,
        "started_at": started,
        "ended_at": ended,
    }


def validate_directory_closure(value: Any, label: str) -> Path:
    require(
        isinstance(value, dict)
        and set(value) == {"path", "file_count", "files", "closure_sha256"},
        f"{label} closure fields differ",
    )
    root = Path(str(value["path"])).expanduser().resolve()
    require(root.is_dir() and not root.is_symlink(), f"{label} root is missing")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        require(not path.is_symlink(), f"{label} contains a symlink: {path}")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    require(
        rows
        and value.get("file_count") == len(rows)
        and value.get("files") == rows
        and value.get("closure_sha256")
        == sha256_bytes(
            json.dumps(
                rows,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ),
        f"{label} closure differs from copied files",
    )
    return root


def m1_cuda_model_contract(source: dict[str, Any]) -> dict[str, Any]:
    lock_path = "scripts/release/configs/runtime_vnext_g08a_m1_cuda.models.lock.json"
    try:
        lock = json.loads(git_file_bytes(source["git_sha"], lock_path))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"invalid source-bound M1 CUDA model lock: {error}") from error
    models = lock.get("models") if isinstance(lock, dict) else None
    require(isinstance(models, list) and len(models) == 1, "M1 CUDA model lock differs")
    model = models[0]
    lane = model.get("lanes", {}).get("cuda") if isinstance(model, dict) else None
    require(
        isinstance(lane, dict)
        and lane.get("hardware_policy") == "cuda-g0-1x-rtx4090"
        and lane.get("repo") == model.get("official_model_id")
        and isinstance(lane.get("revision"), str)
        and GIT_SHA_RE.fullmatch(lane["revision"]) is not None
        and isinstance(lane.get("files"), list)
        and isinstance(lane.get("semantic_source"), dict)
        and lane["semantic_source"].get("repo") == lane.get("repo")
        and lane["semantic_source"].get("revision") == lane.get("revision")
        and isinstance(lane["semantic_source"].get("files"), list),
        "M1 CUDA model lock lane differs",
    )
    unique_files: dict[str, dict[str, Any]] = {}
    ordered_files: list[dict[str, Any]] = []
    for row in [*lane["files"], *lane["semantic_source"]["files"]]:
        require(
            isinstance(row, dict)
            and set(row) == {"path", "sha256", "size_bytes"}
            and isinstance(row.get("path"), str)
            and bool(row["path"])
            and SHA256_RE.fullmatch(str(row.get("sha256"))) is not None
            and isinstance(row.get("size_bytes"), int)
            and not isinstance(row.get("size_bytes"), bool)
            and row["size_bytes"] >= 0,
            "M1 CUDA model lock file identity differs",
        )
        existing = unique_files.get(row["path"])
        require(
            existing is None or existing == row,
            f"M1 CUDA model lock conflicts for {row['path']}",
        )
        if existing is None:
            unique_files[row["path"]] = copy.deepcopy(row)
            ordered_files.append(copy.deepcopy(row))
    return {
        "repo": lane["repo"],
        "revision": lane["revision"],
        "model_files": copy.deepcopy(lane["files"]),
        "semantic_files": copy.deepcopy(lane["semantic_source"]["files"]),
        "unique_files": ordered_files,
        "hardware_policy": lane["hardware_policy"],
        "lock": git_blob_ref(source, lock_path),
    }


def validate_model_lock_validation(
    value: Any,
    *,
    model_argument: str,
    model_contract: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    path = validate_ref(value, f"{label} model lock validation")
    validation = read_json(path, f"{label} model lock validation")
    require(
        validation.get("status") == "pass"
        and validation.get("snapshot_path") == model_argument
        and validation.get("repo") == model_contract["repo"]
        and validation.get("revision") == model_contract["revision"]
        and validation.get("files") == model_contract["unique_files"],
        f"{label} model snapshot validation differs from the source lock",
    )
    return file_ref(path)


def producer_shell_quote(value: str) -> str:
    require(isinstance(value, str), "replay argv contains a non-string value")
    if value and all(character.isalnum() or character in "-_./:" for character in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def producer_command(argv: list[str]) -> str:
    require(argv and all(isinstance(item, str) for item in argv), "replay argv is invalid")
    return " ".join(producer_shell_quote(item) for item in argv)


def parse_product_stdout(path: Path, label: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(isinstance(row, dict), f"{label} stdout line {line_number} is invalid")
            rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"invalid {label} stdout JSONL: {error}") from error
    by_event = {
        event: [row for row in rows if row.get("event") == event]
        for event in ("ready", "user", "assistant", "exit")
    }
    require(
        all(len(by_event[event]) == 1 for event in by_event),
        f"{label} stdout terminal event counts differ",
    )
    ready = by_event["ready"][0]
    user = by_event["user"][0]
    assistant = by_event["assistant"][0]
    exit_row = by_event["exit"][0]
    session_ids = {
        row.get("session_id") for row in (ready, user, assistant, exit_row)
    }
    require(
        len(session_ids) == 1
        and all(isinstance(value, str) and value for value in session_ids)
        and user.get("request_id") == assistant.get("request_id")
        and isinstance(user.get("request_id"), str)
        and bool(user["request_id"])
        and isinstance(assistant.get("content"), str)
        and bool(assistant["content"].strip())
        and isinstance(assistant.get("n_tokens"), int)
        and assistant["n_tokens"] > 0,
        f"{label} stdout execution identity differs",
    )
    return {
        "rows": rows,
        "ready": ready,
        "user": user,
        "assistant": assistant,
        "exit": exit_row,
        "request_id": user["request_id"],
    }


def validate_vast_instance_metadata(
    value: Any,
    *,
    gpu_name: str,
    memory_total_mib: int,
    driver_version: str,
    label: str,
) -> dict[str, Any]:
    path = validate_ref(value, f"{label} Vast instance metadata")
    payload = read_json(path, f"{label} Vast instance metadata")
    instance = payload.get("instances") if isinstance(payload, dict) else None
    if instance is None:
        instance = payload
    require(isinstance(instance, dict), f"{label} Vast instance metadata differs")
    instance_id = instance.get("id")
    vast_gpu_ram = instance.get("gpu_ram")
    require(
        isinstance(instance_id, int)
        and not isinstance(instance_id, bool)
        and instance_id > 0
        and instance.get("cur_state") == "running"
        and instance.get("actual_status") == "running"
        and instance.get("num_gpus") == 1
        and isinstance(instance.get("gpu_name"), str)
        and "4090" in instance["gpu_name"]
        and isinstance(vast_gpu_ram, (int, float))
        and not isinstance(vast_gpu_ram, bool)
        and vast_gpu_ram >= 24_000
        and instance.get("driver_version") == driver_version
        and "4090" in gpu_name
        and memory_total_mib >= 24_000
        and abs(float(vast_gpu_ram) - memory_total_mib) <= 256,
        f"{label} Vast/GPU hardware class or driver differs",
    )
    return {
        "provider": "vast",
        "instance_id": instance_id,
        "instance_metadata": file_ref(path),
        "hardware_class": "1x-rtx4090-24gb",
        "vast_gpu_name": instance["gpu_name"],
        "vast_gpu_ram_mib": vast_gpu_ram,
    }


def derive_cuda_full_profile_metrics(
    events: list[dict[str, Any]], check_id: str
) -> dict[str, Any]:
    try:
        import runtime_vnext_r2_profile_collector as profile_collector
    except ImportError as error:
        raise QualificationError("R2 profile collector validator is unavailable") from error
    try:
        stage = profile_collector.calculate_stage_coverage(events)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise QualificationError(f"{check_id} stage coverage is invalid: {error}") from error
    require(
        stage.get("formal_coverage_eligible") is True,
        f"{check_id} stage timing lacks a bounded clock conversion",
    )
    stage_coverage = float(stage.get("coverage", -1.0))
    decode_wall = stage.get("decode_wall_ns")
    conversion = stage.get("clock_conversion")
    require(
        isinstance(decode_wall, int)
        and decode_wall > 0
        and isinstance(conversion, dict)
        and isinstance(conversion.get("relative_max_error_nanos"), int),
        f"{check_id} clock conversion evidence is invalid",
    )
    clock_error = conversion["relative_max_error_nanos"] / decode_wall

    def attributes(event: dict[str, Any]) -> dict[str, Any]:
        value = event.get("attributes")
        return value if isinstance(value, dict) else {}

    native_rows = [
        event
        for event in events
        if event.get("phase") == "vnext.device_native_work"
    ]
    timing_events = [
        event
        for event in events
        if event.get("phase")
        in {"vnext.device_native_work", "vnext.device_execution_span"}
    ]
    require(native_rows and timing_events, f"{check_id} lacks CUDA timing rows")
    measured_span_ranges: dict[str, list[tuple[int, int]]] = {}
    for event in timing_events:
        if event.get("phase") != "vnext.device_execution_span":
            continue
        row_attributes = attributes(event)
        shape = event.get("shape")
        if (
            row_attributes.get("device_timing_status") != "measured"
            or not isinstance(shape, dict)
        ):
            continue
        fingerprint = row_attributes.get("physical_submission_fingerprint")
        start = shape.get("start_command_index")
        end = shape.get("end_command_index")
        if (
            isinstance(fingerprint, str)
            and fingerprint
            and isinstance(start, int)
            and not isinstance(start, bool)
            and isinstance(end, int)
            and not isinstance(end, bool)
            and 0 <= start < end
        ):
            measured_span_ranges.setdefault(fingerprint, []).append((start, end))

    total_dispatches = 0
    timed_dispatches = 0
    for event in native_rows:
        row_attributes = attributes(event)
        shape = event.get("shape")
        require(isinstance(shape, dict), f"{check_id} native timing row lacks shape")
        compute = shape.get("physical_compute_dispatch_count", 0)
        transfer = shape.get("physical_transfer_command_count", 0)
        require(
            isinstance(compute, int)
            and not isinstance(compute, bool)
            and compute >= 0
            and isinstance(transfer, int)
            and not isinstance(transfer, bool)
            and transfer >= 0,
            f"{check_id} native dispatch count is invalid",
        )
        dispatches = compute + transfer
        if dispatches <= 0:
            continue
        total_dispatches += dispatches
        covered = row_attributes.get("device_timing_status") == "measured"
        if not covered:
            fingerprint = row_attributes.get("physical_submission_fingerprint")
            command_index = shape.get("command_index")
            if isinstance(fingerprint, str) and isinstance(command_index, int):
                covered = any(
                    start <= command_index < end
                    for start, end in measured_span_ranges.get(fingerprint, [])
                )
        if covered:
            timed_dispatches += dispatches
    require(total_dispatches > 0, f"{check_id} has no CUDA dispatches")
    dispatch_coverage = timed_dispatches / total_dispatches

    total_device_ns = 0
    attributed_device_ns = 0
    measured_rows = 0
    for event in timing_events:
        row_attributes = attributes(event)
        shape = event.get("shape")
        require(isinstance(shape, dict), f"{check_id} CUDA timing row lacks shape")
        if row_attributes.get("device_timing_status") != "measured":
            continue
        elapsed = shape.get("device_elapsed_ns")
        require(
            isinstance(elapsed, int)
            and not isinstance(elapsed, bool)
            and elapsed > 0,
            f"{check_id} measured CUDA row lacks positive elapsed time",
        )
        measured_rows += 1
        total_device_ns += elapsed
        has_native_mapping = (
            event.get("phase") == "vnext.device_native_work"
            and isinstance(row_attributes.get("native_op_id"), str)
            and bool(row_attributes["native_op_id"])
            and (
                row_attributes.get("attribution_scope") != "node"
                or all(
                    isinstance(row_attributes.get(field), str)
                    and bool(row_attributes[field])
                    for field in ("node_id", "operation_id", "provider_id")
                )
            )
        )
        intervals = event.get("backend_detail")
        intervals = (
            intervals.get("device_intervals")
            if isinstance(intervals, dict)
            else None
        )
        require(
            isinstance(intervals, list),
            f"{check_id} measured CUDA row lacks device subwork intervals",
        )
        interval_rows: list[tuple[int, int]] = []
        for interval in intervals:
            require(isinstance(interval, dict), f"{check_id} subwork interval is invalid")
            start = interval.get("start_offset_ns")
            end = interval.get("end_offset_ns")
            require(
                isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(end, int)
                and not isinstance(end, bool)
                and 0 <= start < end
                and isinstance(interval.get("subwork_id"), str)
                and bool(interval["subwork_id"]),
                f"{check_id} device subwork interval is invalid",
            )
            interval_rows.append((start, end))
        try:
            interval_ns = profile_collector.validate_device_interval_contract(
                interval_rows,
                elapsed_ns=elapsed,
                context=f"{check_id} measured CUDA row",
            )
        except profile_collector.CollectorError as error:
            raise QualificationError(str(error)) from error
        if has_native_mapping or interval_ns > 0:
            attributed_device_ns += elapsed
    require(
        measured_rows > 0 and total_device_ns > 0,
        f"{check_id} has no measured CUDA device timing",
    )
    attribution_coverage = attributed_device_ns / total_device_ns
    require(
        stage_coverage >= MIN_STAGE_COVERAGE
        and clock_error <= MAX_CLOCK_RELATIVE_ERROR
        and dispatch_coverage >= MIN_DISPATCH_TIMING_COVERAGE
        and attribution_coverage >= MIN_DEVICE_ATTRIBUTION_COVERAGE,
        f"{check_id} profile timing thresholds failed: "
        f"stage={stage_coverage:.6f}, clock={clock_error:.6f}, "
        f"dispatch={dispatch_coverage:.6f}, attribution={attribution_coverage:.6f}",
    )
    return {
        "stage_coverage": stage_coverage,
        "minimum_stage_coverage": MIN_STAGE_COVERAGE,
        "clock_relative_error_fraction": clock_error,
        "maximum_clock_relative_error_fraction": MAX_CLOCK_RELATIVE_ERROR,
        "dispatch_timing_coverage": dispatch_coverage,
        "minimum_dispatch_timing_coverage": MIN_DISPATCH_TIMING_COVERAGE,
        "device_attribution_coverage": attribution_coverage,
        "minimum_device_attribution_coverage": MIN_DEVICE_ATTRIBUTION_COVERAGE,
        "measured_device_timing_row_count": measured_rows,
        "timed_dispatch_count": timed_dispatches,
        "total_dispatch_count": total_dispatches,
    }


def _validate_raw_execution_v1(
    path: Path,
    *,
    check_id: str,
    entrypoint: str,
    profile_detail: str,
    build_source: dict[str, Any],
    binary_sha256: str,
    require_timing_metrics: bool = False,
    expected_build_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = read_json(path, f"{check_id} raw execution")
    require(
        set(raw)
        == {
            "schema_version",
            "artifact_type",
            "source_git_sha",
            "source_tree_sha",
            "backend",
            "binary_copy",
            "executed_binary_path",
            "build_receipt",
            "profile_jsonl",
            "command",
            "http",
            "harness",
        },
        f"{check_id} raw execution field set differs",
    )
    require(
        raw.get("schema_version") == 1
        and raw.get("artifact_type") == RAW_EVIDENCE_TYPE
        and raw.get("source_git_sha") == build_source["git_sha"]
        and raw.get("source_tree_sha") == build_source["git_tree_sha"]
        and raw.get("backend") == "cuda",
        f"{check_id} raw execution identity differs",
    )
    raw_binary_path = validate_ref(
        raw.get("binary_copy"), f"{check_id} raw binary copy"
    )
    raw_binary = file_ref(raw_binary_path)
    raw_build_receipt_path = validate_ref(
        raw.get("build_receipt"), f"{check_id} raw build receipt"
    )
    raw_build_receipt = read_json(
        raw_build_receipt_path, f"{check_id} raw build receipt"
    )
    if expected_build_receipt is not None:
        require(
            file_ref(raw_build_receipt_path) == expected_build_receipt,
            f"{check_id} raw execution used a different build receipt",
        )
    require(
        raw_binary["sha256"] == binary_sha256
        and raw_build_receipt.get("artifact_type")
        == "runtime_vnext_candidate_build_receipt"
        and raw_build_receipt.get("status") == "pass"
        and raw_build_receipt.get("backend") == "cuda"
        and raw_build_receipt.get("binary_sha256") == binary_sha256
        and raw_build_receipt.get("source_git_sha") == build_source["git_sha"]
        and raw_build_receipt.get("source_tree_sha")
        == build_source["git_tree_sha"]
        and isinstance(raw.get("executed_binary_path"), str)
        and Path(raw["executed_binary_path"]).name == "ferrum",
        f"{check_id} raw execution used a different binary",
    )
    profile_path = validate_ref(
        raw.get("profile_jsonl"), f"{check_id} profile JSONL"
    )
    reusable_spans = 0
    reusable_fingerprints: set[str] = set()
    verification_sequence_completed: set[str] = set()
    verification_request_completed: set[str] = set()
    verification_eager_participants: set[str] = set()
    profile_startup_error_count = 0
    profile_events: list[dict[str, Any]] = []
    try:
        with profile_path.open("r", encoding="utf-8") as profile_handle:
            for line_number, line in enumerate(profile_handle, 1):
                profile_startup_error_count += line.count(OLD_STARTUP_ERROR)
                if not line.strip():
                    continue
                event = json.loads(line)
                require(
                    isinstance(event, dict),
                    f"{check_id} profile event {line_number} is not an object",
                )
                profile_events.append(event)
                attributes = event.get("attributes")
                request_id = event.get("request_id")
                if (
                    profile_detail == "verify"
                    and event.get("entrypoint") == entrypoint
                    and isinstance(request_id, str)
                    and request_id.startswith("request.startup.")
                    and isinstance(attributes, dict)
                    and attributes.get("profile_detail") == "verify"
                    and isinstance(attributes.get("backend_device"), str)
                    and attributes["backend_device"].startswith("CUDA(")
                ):
                    phase = event.get("phase")
                    if phase in {
                        "vnext.sequence_completed",
                        "vnext.request_completed",
                    }:
                        expected_kind = phase.removeprefix("vnext.")
                        require(
                            event.get("status") == "ok"
                            and attributes.get("execution_event_kind")
                            == expected_kind
                            and attributes.get("execution_request_origin")
                            == "startup"
                            and attributes.get("execution_request_id") == request_id
                            and isinstance(
                                attributes.get("completed_sequence_fingerprint"), str
                            )
                            and SHA256_RE.fullmatch(
                                attributes["completed_sequence_fingerprint"]
                            )
                            is not None,
                            f"{check_id} startup verification terminal event is invalid",
                        )
                        if phase == "vnext.sequence_completed":
                            verification_sequence_completed.add(request_id)
                        else:
                            verification_request_completed.add(request_id)
                    elif phase == "vnext.device_native_work":
                        participants = attributes.get("participant_request_ids")
                        require(
                            event.get("status") == "diagnostic_only"
                            and attributes.get("execution_path") == "eager"
                            and attributes.get("device_timing_span_kind")
                            == "eager_command"
                            and isinstance(participants, list)
                            and participants
                            and all(isinstance(item, str) for item in participants)
                            and request_id in participants,
                            f"{check_id} startup verification native work is invalid",
                        )
                        verification_eager_participants.update(
                            item
                            for item in participants
                            if item.startswith("request.startup.")
                        )
                if not (
                    event.get("phase") == "vnext.device_execution_span"
                    and event.get("entrypoint") == entrypoint
                    and isinstance(request_id, str)
                    and request_id.startswith("request.startup.")
                    and isinstance(attributes, dict)
                    and attributes.get("device_timing_span_kind")
                    == "reusable_executable"
                    and attributes.get("execution_path") == "replayed"
                    and attributes.get("profile_detail") == profile_detail
                    and isinstance(attributes.get("backend_device"), str)
                    and attributes["backend_device"].startswith("CUDA(")
                ):
                    continue
                fingerprint = attributes.get("reusable_executable_fingerprint")
                require(
                    isinstance(fingerprint, str)
                    and SHA256_RE.fullmatch(fingerprint) is not None,
                    f"{check_id} startup reusable span fingerprint is invalid",
                )
                reusable_spans += 1
                reusable_fingerprints.add(fingerprint)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(
            f"invalid {check_id} profile JSONL {profile_path}: {error}"
        ) from error
    if profile_detail == "full":
        require(
            reusable_spans > 0 and reusable_fingerprints,
            f"{check_id} did not reach startup reusable executable capture/replay",
        )
    else:
        require(
            verification_request_completed
            and verification_request_completed == verification_sequence_completed
            and verification_request_completed == verification_eager_participants,
            f"{check_id} did not complete CUDA startup verification work",
        )

    command = raw.get("command")
    require(
        isinstance(command, dict)
        and set(command) == {"argv", "returncode", "stdout", "stderr"},
        f"{check_id} raw command field set differs",
    )
    argv = direct_product_command(
        command.get("argv"),
        check_id=check_id,
        entrypoint=entrypoint,
        profile_detail=profile_detail,
    )
    require(
        argv[0] == raw["executed_binary_path"],
        f"{check_id} command differs from its recorded executed binary path",
    )
    command_rc_value = command.get("returncode")
    if entrypoint == "run":
        command_rc = strict_process_returncode(
            command_rc_value, f"{check_id} product command"
        )
    else:
        require(
            command_rc_value is None
            or (
                isinstance(command_rc_value, int)
                and not isinstance(command_rc_value, bool)
            ),
            f"{check_id} serve command return code is invalid",
        )
        command_rc = command_rc_value
    stdout_path = validate_ref(command.get("stdout"), f"{check_id} stdout")
    stderr_path = validate_ref(command.get("stderr"), f"{check_id} stderr")
    evidence_refs: dict[str, Any] = {
        "build_receipt": file_ref(raw_build_receipt_path),
        "profile_jsonl": file_ref(profile_path),
        "stdout": file_ref(stdout_path),
        "stderr": file_ref(stderr_path),
    }
    scanned_paths = [stdout_path, stderr_path]

    http = raw.get("http")
    response: dict[str, Any] | None = None
    if entrypoint == "run":
        require(http is None, f"{check_id} run evidence unexpectedly has HTTP data")
        product = {"success": command_rc == 0, "command_rc": command_rc}
        derived_http = None
    else:
        require(
            isinstance(http, dict)
            and set(http) == {"status_code", "request_body", "response_body"},
            f"{check_id} HTTP evidence field set differs",
        )
        status_code = http.get("status_code")
        require(
            isinstance(status_code, int) and not isinstance(status_code, bool),
            f"{check_id} HTTP status is invalid",
        )
        request_path = validate_ref(
            http.get("request_body"), f"{check_id} HTTP request body"
        )
        request = read_json(request_path, f"{check_id} HTTP request body")
        response_path = validate_ref(
            http.get("response_body"), f"{check_id} HTTP response body"
        )
        response = read_json(response_path, f"{check_id} HTTP response body")
        usage = response.get("usage")
        completion_tokens = (
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        require(
            isinstance(completion_tokens, int)
            and not isinstance(completion_tokens, bool),
            f"{check_id} HTTP completion token count is invalid",
        )
        product = {
            "success": status_code == 200 and completion_tokens > 0,
            "http_status": status_code,
            "completion_tokens": completion_tokens,
        }
        derived_http = {
            "status_code": status_code,
            "request_body": file_ref(request_path),
            "response_body": file_ref(response_path),
        }
        evidence_refs["http_request_body"] = file_ref(request_path)
        evidence_refs["http_response_body"] = file_ref(response_path)
        scanned_paths.extend([request_path, response_path])

    harness = raw.get("harness")
    require(
        isinstance(harness, dict)
        and set(harness) == {"returncode", "diagnostic"},
        f"{check_id} raw harness field set differs",
    )
    harness_rc = strict_process_returncode(
        harness.get("returncode"), f"{check_id} harness"
    )
    diagnostic = harness.get("diagnostic")
    if harness_rc == 0:
        require(
            diagnostic is None,
            f"{check_id} passing harness unexpectedly has a diagnostic",
        )
        harness_result = {"status": "pass", "failure_class": None}
    else:
        diagnostic_path = validate_ref(
            diagnostic, f"{check_id} harness diagnostic"
        )
        evidence_refs["harness_diagnostic"] = file_ref(diagnostic_path)
        scanned_paths.append(diagnostic_path)
        choices = response.get("choices") if isinstance(response, dict) else None
        message = (
            choices[0].get("message")
            if isinstance(choices, list)
            and choices
            and isinstance(choices[0], dict)
            else None
        )
        content = message.get("content") if isinstance(message, dict) else object()
        reasoning = message.get("reasoning") if isinstance(message, dict) else None
        finish_reason = (
            choices[0].get("finish_reason")
            if isinstance(choices, list)
            and choices
            and isinstance(choices[0], dict)
            else None
        )
        request_max_tokens = request.get("max_tokens")
        template_kwargs = request.get("chat_template_kwargs")
        thinking_disabled = (
            isinstance(template_kwargs, dict)
            and template_kwargs.get("enable_thinking") is False
        )
        diagnostic_text = diagnostic_path.read_text(
            encoding="utf-8", errors="replace"
        ).lower()
        fatal_markers = (
            "out of memory",
            "cuda oom",
            "panic",
            "segmentation fault",
            "server timed out",
            "server timeout",
            "server killed",
        )
        require(
            entrypoint == "serve"
            and product["success"] is True
            and command_rc is None
            and isinstance(message, dict)
            and content in (None, "")
            and isinstance(reasoning, str)
            and bool(reasoning.strip())
            and finish_reason == "length"
            and isinstance(request_max_tokens, int)
            and not isinstance(request_max_tokens, bool)
            and request_max_tokens == product["completion_tokens"]
            and not thinking_disabled,
            f"{check_id} harness failure is not derivable as non-target configuration",
        )
        require(
            not any(marker in diagnostic_text for marker in fatal_markers)
            and not any(
                marker
                in evidence_path.read_text(
                    encoding="utf-8", errors="replace"
                ).lower()
                for evidence_path in scanned_paths
                for marker in fatal_markers
            ),
            f"{check_id} harness diagnostic contains a product-fatal marker",
        )
        harness_result = {
            "status": "failed",
            "failure_class": "non_target_configuration",
        }

    startup_error_count = profile_startup_error_count + sum(
        evidence_path.read_text(encoding="utf-8", errors="replace").count(
            OLD_STARTUP_ERROR
        )
        for evidence_path in scanned_paths
    )
    target_signal = {
        "startup_catalog_error_count": startup_error_count,
        "startup_reusable_execution_span_count": reusable_spans,
        "startup_reusable_program_fingerprint_count": len(reusable_fingerprints),
        "startup_verification_sequence_completed_count": len(
            verification_sequence_completed
        ),
        "startup_verification_request_completed_count": len(
            verification_request_completed
        ),
        "startup_verification_eager_participant_count": len(
            verification_eager_participants
        ),
    }
    if profile_detail == "full" and require_timing_metrics:
        target_signal.update(
            derive_cuda_full_profile_metrics(profile_events, check_id)
        )
    return {
        "raw_execution": file_ref(path),
        "source": {
            "git_sha": raw["source_git_sha"],
            "git_tree_sha": raw["source_tree_sha"],
            "dirty": False,
        },
        "binary": raw_binary,
        "command": {"argv": argv, "returncode": command_rc},
        "http": derived_http,
        "product": product,
        "target_signal": target_signal,
        "harness": harness_result,
        "artifacts": evidence_refs,
    }


def _validate_raw_execution_v2(
    path: Path,
    *,
    check_id: str,
    entrypoint: str,
    profile_detail: str,
    build_source: dict[str, Any],
    binary_sha256: str,
    expected_build_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    require(
        entrypoint == "run" and profile_detail == "full",
        f"{check_id} execution-closure schema only supports run/full",
    )
    raw = read_json(path, f"{check_id} raw execution")
    require(
        set(raw)
        == {
            "schema_version",
            "artifact_type",
            "source_git_sha",
            "source_tree_sha",
            "backend",
            "vast_instance_metadata",
            "binary_copy",
            "executed_binary",
            "build_receipt",
            "product_receipt",
            "product_stdout",
            "product_stderr",
            "hardware_receipt",
            "hardware_stdout",
            "hardware_stderr",
            "execution_outputs",
            "model_identity",
            "model_lock_validation",
        },
        f"{check_id} raw execution-closure field set differs",
    )
    require(
        raw.get("schema_version") == 2
        and raw.get("artifact_type") == RAW_EVIDENCE_TYPE
        and raw.get("source_git_sha") == build_source["git_sha"]
        and raw.get("source_tree_sha") == build_source["git_tree_sha"]
        and raw.get("backend") == "cuda",
        f"{check_id} raw execution-closure identity differs",
    )

    binary_path = validate_ref(raw.get("binary_copy"), f"{check_id} binary copy")
    binary = file_ref(binary_path)
    executed_binary = raw.get("executed_binary")
    require(
        isinstance(executed_binary, dict)
        and set(executed_binary) == {"path", "sha256", "size_bytes"}
        and isinstance(executed_binary.get("path"), str)
        and Path(executed_binary["path"]).name == "ferrum"
        and executed_binary.get("sha256") == binary.get("sha256") == binary_sha256
        and executed_binary.get("size_bytes") == binary.get("size_bytes"),
        f"{check_id} executed binary identity differs from its copied bytes",
    )
    build_receipt_path = validate_ref(
        raw.get("build_receipt"), f"{check_id} build receipt"
    )
    build_receipt = read_json(build_receipt_path, f"{check_id} build receipt")
    require(
        (expected_build_receipt is None or file_ref(build_receipt_path) == expected_build_receipt)
        and build_receipt.get("artifact_type")
        == "runtime_vnext_candidate_build_receipt"
        and build_receipt.get("status") == "pass"
        and build_receipt.get("returncode") == 0
        and build_receipt.get("binary_sha256") == binary_sha256
        and build_receipt.get("source_git_sha") == build_source["git_sha"]
        and build_receipt.get("source_tree_sha") == build_source["git_tree_sha"],
        f"{check_id} build/binary closure differs",
    )

    product_run = validate_bounded_execution_receipt(
        raw.get("product_receipt"),
        raw.get("product_stdout"),
        raw.get("product_stderr"),
        f"{check_id} product",
    )
    argv = direct_product_command(
        product_run["command"],
        check_id=check_id,
        entrypoint="run",
        profile_detail="full",
    )
    require(
        argv[0] == executed_binary["path"]
        and len(argv) >= 3
        and exact_flag_value(argv, "--backend", check_id) == "cuda"
        and exact_flag_value(argv, "--profile-sample-rate", check_id)
        in {"1", "1.0"}
        and exact_flag_value(argv, "--output-format", check_id) == "jsonl",
        f"{check_id} product command identity differs",
    )

    outputs = raw.get("execution_outputs")
    require(
        isinstance(outputs, dict)
        and set(outputs) == {"profile_jsonl", "effective_config", "request_dump"},
        f"{check_id} execution output set differs",
    )
    profile_output = outputs["profile_jsonl"]
    config_output = outputs["effective_config"]
    dump_output = outputs["request_dump"]
    require(
        isinstance(profile_output, dict)
        and set(profile_output) == {"executed_path", "copy"}
        and isinstance(config_output, dict)
        and set(config_output) == {"executed_path", "copy"}
        and isinstance(dump_output, dict)
        and set(dump_output) == {"executed_path", "copy"},
        f"{check_id} execution output descriptors differ",
    )
    profile_path = validate_ref(
        profile_output["copy"], f"{check_id} profile JSONL"
    )
    config_path = validate_ref(
        config_output["copy"], f"{check_id} effective config"
    )
    dump_root = validate_directory_closure(
        dump_output["copy"], f"{check_id} request dump"
    )
    require(
        exact_flag_value(argv, "--profile-jsonl", check_id)
        == profile_output["executed_path"]
        and exact_flag_value(argv, "--effective-config-json", check_id)
        == config_output["executed_path"]
        and exact_flag_value(argv, "--request-dump-dir", check_id)
        == dump_output["executed_path"],
        f"{check_id} command/output path binding differs",
    )
    effective_config = read_json(config_path, f"{check_id} effective config")
    require(
        effective_config.get("backend") == "cuda",
        f"{check_id} effective config backend differs",
    )

    model_contract = m1_cuda_model_contract(build_source)
    model_argument = argv[2]
    semantic_argument = exact_flag_value(argv, "--semantic-source", check_id)
    tokenizer_argument = exact_flag_value(argv, "--tokenizer-source", check_id)
    expected_model_identity = {
        "model_argument": model_argument,
        "semantic_argument": semantic_argument,
        "repo": model_contract["repo"],
        "revision": model_contract["revision"],
        "model_files": model_contract["model_files"],
        "semantic_files": model_contract["semantic_files"],
        "lock": model_contract["lock"],
    }
    require(
        model_argument == semantic_argument == tokenizer_argument
        and raw.get("model_identity") == expected_model_identity,
        f"{check_id} source-bound M1 CUDA model identity differs",
    )
    model_lock_validation = validate_model_lock_validation(
        raw.get("model_lock_validation"),
        model_argument=model_argument,
        model_contract=model_contract,
        label=check_id,
    )

    stdout = parse_product_stdout(product_run["stdout"], check_id)
    ready = stdout["ready"]
    product_request_id = stdout["request_id"]
    require(
        ready.get("requested_model") == model_argument
        and ready.get("resolved_model") == model_contract["repo"]
        and ready.get("model") == model_contract["repo"]
        and ready.get("backend") == "CUDA(0)",
        f"{check_id} runtime model/device identity differs",
    )

    try:
        import request_replay_bundle_gate as replay_gate

        bundles = replay_gate.validate_bundle_root(dump_root)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise QualificationError(f"{check_id} request replay bundle failed: {error}") from error
    require(len(bundles) == 1, f"{check_id} must contain exactly one request bundle")
    bundle_root = Path(str(bundles[0]["bundle_dir"])).resolve()
    request = read_json(bundle_root / "request.json", f"{check_id} request bundle")
    runtime_config = read_json(
        bundle_root / "runtime_effective_config.json",
        f"{check_id} request runtime config",
    )
    backend_selection = read_json(
        bundle_root / "backend_selection.json",
        f"{check_id} request backend selection",
    )
    sampling = read_json(
        bundle_root / "sampling_params.json", f"{check_id} request sampling"
    )
    replay_command = read_json(
        bundle_root / "replay.command.json", f"{check_id} request replay command"
    )
    replay_argv = replay_command.get("argv")
    replay_sample_rate = (
        exact_flag_value(replay_argv, "--profile-sample-rate", check_id)
        if isinstance(replay_argv, list)
        else ""
    )
    expected_bundle_dir = str(
        Path(str(dump_output["executed_path"])) / product_request_id
    )
    expected_replay_argv = [
        "cargo",
        "run",
        "-p",
        "ferrum-cli",
        "--",
        "run",
        "synthetic/no-weight",
        "--profile-detail",
        "full",
        "--profile-sample-rate",
        replay_sample_rate,
    ]
    for flag, field in (
        ("--profile-jsonl", "profile_jsonl"),
        ("--memory-profile-jsonl", "memory_profile_jsonl"),
        ("--scheduler-trace-jsonl", "scheduler_trace_jsonl"),
        ("--request-dump-dir", "request_dump_dir"),
    ):
        value = runtime_config.get(field)
        if value is not None:
            require(
                isinstance(value, str) and value,
                f"{check_id} request runtime {field} is invalid",
            )
            expected_replay_argv.extend([flag, value])
    engine_replay = replay_command.get("engine_replay")
    expected_engine_argv = [
        "cargo",
        "run",
        "-p",
        "ferrum-cli",
        "--",
        "replay-bundle",
        expected_bundle_dir,
        "--out",
        str(Path(expected_bundle_dir) / "engine_replay"),
        "--json",
    ]
    require(
        bundles[0].get("request_id") == product_request_id
        and request.get("request_id") == product_request_id
        and request.get("schema_version") == 1
        and request.get("entrypoint") == "run"
        and request.get("backend") == "actual"
        and request.get("actual_model_smoke") is True
        and request.get("sanitized") is True
        and request.get("l0_only") is False
        and request.get("profile_detail") == "full"
        and request.get("profile_sample_rate") == 1.0
        and request.get("model") == model_argument
        and runtime_config.get("request_id") == product_request_id
        and runtime_config.get("schema_version") == 1
        and runtime_config.get("entrypoint") == "run"
        and runtime_config.get("profile_detail") == "full"
        and runtime_config.get("profile_sample_rate") == 1.0
        and runtime_config.get("profile_jsonl") == profile_output["executed_path"]
        and runtime_config.get("request_dump_dir") == dump_output["executed_path"]
        and runtime_config.get("sanitized") is True
        and backend_selection.get("request_id") == product_request_id
        and backend_selection.get("schema_version") == 1
        and backend_selection.get("backend") == "actual"
        and backend_selection.get("actual_model_smoke") is True
        and backend_selection.get("model") == model_argument
        and sampling.get("request_id") == product_request_id
        and replay_argv == expected_replay_argv
        and replay_sample_rate in {"1", "1.0"}
        and replay_command.get("schema_version") == 1
        and replay_command.get("request_id") == product_request_id
        and replay_command.get("entrypoint") == "run"
        and replay_command.get("sanitized") is True
        and replay_command.get("bundle_dir") == expected_bundle_dir
        and replay_command.get("command") == producer_command(expected_replay_argv)
        and request.get("replay_command") == replay_command.get("command")
        and isinstance(engine_replay, dict)
        and engine_replay.get("mode") == "bundle_offline"
        and engine_replay.get("requires_http_server") is False
        and engine_replay.get("argv") == expected_engine_argv
        and engine_replay.get("command") == producer_command(expected_engine_argv),
        f"{check_id} request/model/backend closure differs",
    )
    if "--prompt" in argv:
        require(
            stdout["user"].get("content")
            == exact_flag_value(argv, "--prompt", check_id),
            f"{check_id} product prompt differs from stdout user event",
        )
    sampling_params = sampling.get("sampling_params")
    if isinstance(sampling_params, dict) and "max_tokens" in sampling_params:
        require(
            str(sampling_params["max_tokens"])
            == exact_flag_value(argv, "--max-tokens", check_id),
            f"{check_id} request max_tokens differs from product command",
        )

    hardware_run = validate_bounded_execution_receipt(
        raw.get("hardware_receipt"),
        raw.get("hardware_stdout"),
        raw.get("hardware_stderr"),
        f"{check_id} hardware",
    )
    require(
        hardware_run["command"]
        == [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
        and hardware_run["ended_at"] <= product_run["started_at"],
        f"{check_id} hardware probe command/time differs",
    )
    hardware_lines = [
        line.strip()
        for line in hardware_run["stdout"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(hardware_lines) == 1, f"{check_id} requires exactly one CUDA GPU")
    hardware_columns = [item.strip() for item in hardware_lines[0].split(",")]
    require(len(hardware_columns) == 5, f"{check_id} hardware row differs")
    try:
        gpu_index = int(hardware_columns[0])
        memory_total_mib = int(hardware_columns[3])
    except ValueError as error:
        raise QualificationError(f"{check_id} hardware numeric identity differs") from error
    witness_hardware = {
        "policy_id": model_contract["hardware_policy"],
        "gpu_count": 1,
        "index": gpu_index,
        "name": hardware_columns[1],
        "uuid": hardware_columns[2],
        "memory_total_mib": memory_total_mib,
        "driver_version": hardware_columns[4],
    }
    require(
        gpu_index == 0
        and "4090" in witness_hardware["name"]
        and re.fullmatch(
            r"GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}",
            witness_hardware["uuid"],
        )
        is not None
        and memory_total_mib >= 24_000
        and bool(witness_hardware["driver_version"]),
        f"{check_id} exact hardware identity is invalid",
    )
    witness_hardware.update(
        validate_vast_instance_metadata(
            raw.get("vast_instance_metadata"),
            gpu_name=witness_hardware["name"],
            memory_total_mib=memory_total_mib,
            driver_version=witness_hardware["driver_version"],
            label=check_id,
        )
    )

    profile_events: list[dict[str, Any]] = []
    profile_text = profile_path.read_text(encoding="utf-8")
    try:
        for line_number, line in enumerate(profile_text.splitlines(), 1):
            if not line.strip():
                continue
            event = json.loads(line)
            require(isinstance(event, dict), f"{check_id} profile line {line_number} is invalid")
            profile_events.append(event)
    except json.JSONDecodeError as error:
        raise QualificationError(f"{check_id} profile JSONL is invalid: {error}") from error
    require(profile_events, f"{check_id} profile JSONL is empty")
    event_ids = [event.get("event_id") for event in profile_events]
    require(
        all(isinstance(event_id, str) and event_id for event_id in event_ids)
        and len(set(event_ids)) == len(event_ids)
        and not any(event.get("status") == "failure" for event in profile_events),
        f"{check_id} profile event identity/status differs",
    )
    product_events: list[dict[str, Any]] = []
    reusable_fingerprints: set[str] = set()
    reusable_spans = 0
    observed_devices: set[str] = set()
    for event in profile_events:
        attributes = event.get("attributes")
        attributes = attributes if isinstance(attributes, dict) else {}
        require(
            event.get("entrypoint") == "run"
            and event.get("model") == model_contract["repo"]
            and attributes.get("profile_detail") == "full",
            f"{check_id} profile contains a foreign entrypoint/model/mode",
        )
        timestamp = parse_utc(event.get("timestamp"), f"{check_id} profile event")
        require(
            product_run["started_at"] <= timestamp <= product_run["ended_at"],
            f"{check_id} profile event lies outside the bounded product execution",
        )
        backend_device = attributes.get("backend_device")
        if backend_device is not None:
            require(isinstance(backend_device, str), f"{check_id} backend_device is invalid")
            observed_devices.add(backend_device)
        joined_ids = {
            value
            for value in (
                event.get("request_id"),
                event.get("correlation_id"),
                attributes.get("execution_request_id"),
            )
            if isinstance(value, str) and value
        }
        is_product_event = product_request_id in joined_ids
        if is_product_event:
            product_events.append(event)
        else:
            require(
                joined_ids
                and all(value.startswith("request.startup.") for value in joined_ids),
                f"{check_id} profile contains an unrelated request closure",
            )
        if attributes.get("device_timing_status") == "measured":
            require(
                is_product_event,
                f"{check_id} measured timing row is unrelated to the product request",
            )
        if (
            event.get("phase") == "vnext.device_execution_span"
            and any(value.startswith("request.startup.") for value in joined_ids)
            and attributes.get("device_timing_span_kind") == "reusable_executable"
            and attributes.get("execution_path") == "replayed"
        ):
            fingerprint = attributes.get("reusable_executable_fingerprint")
            require(
                isinstance(fingerprint, str)
                and SHA256_RE.fullmatch(fingerprint) is not None,
                f"{check_id} startup reusable fingerprint is invalid",
            )
            reusable_spans += 1
            reusable_fingerprints.add(fingerprint)
    require(
        observed_devices == {f"CUDA({gpu_index})"}
        and reusable_spans > 0
        and reusable_fingerprints,
        f"{check_id} hardware/reusable profile closure differs",
    )
    terminal_rows = [
        event
        for event in product_events
        if event.get("phase") == "actual_run_generation"
    ]
    accepted_rows = [
        event
        for event in product_events
        if event.get("phase") == "vnext.request_accepted"
    ]
    require(
        len(terminal_rows) == 1
        and len(accepted_rows) == 1
        and terminal_rows[0].get("attributes", {}).get("execution_request_id")
        == product_request_id,
        f"{check_id} product request/profile terminal join differs",
    )
    plan_ids = {
        event.get("attributes", {}).get("plan_id")
        for event in product_events
        if isinstance(event.get("attributes"), dict)
        and event["attributes"].get("plan_id") is not None
    }
    require(
        len(plan_ids) == 1 and all(isinstance(item, str) and item for item in plan_ids),
        f"{check_id} product profile plan identity differs",
    )
    target_signal = {
        "startup_catalog_error_count": profile_text.count(OLD_STARTUP_ERROR)
        + product_run["stdout"].read_text(encoding="utf-8", errors="replace").count(OLD_STARTUP_ERROR)
        + product_run["stderr"].read_text(encoding="utf-8", errors="replace").count(OLD_STARTUP_ERROR),
        "startup_reusable_execution_span_count": reusable_spans,
        "startup_reusable_program_fingerprint_count": len(reusable_fingerprints),
        "startup_verification_sequence_completed_count": 0,
        "startup_verification_request_completed_count": 0,
        "startup_verification_eager_participant_count": 0,
        "product_profile_event_count": len(product_events),
        "product_request_id_sha256": sha256_bytes(product_request_id.encode("utf-8")),
    }
    target_signal.update(derive_cuda_full_profile_metrics(product_events, check_id))
    artifacts = {
        "build_receipt": file_ref(build_receipt_path),
        "product_receipt": product_run["receipt"],
        "profile_jsonl": file_ref(profile_path),
        "stdout": file_ref(product_run["stdout"]),
        "stderr": file_ref(product_run["stderr"]),
        "effective_config": file_ref(config_path),
        "model_lock_validation": model_lock_validation,
        "hardware_receipt": hardware_run["receipt"],
        "hardware_stdout": file_ref(hardware_run["stdout"]),
        "hardware_stderr": file_ref(hardware_run["stderr"]),
    }
    return {
        "raw_execution": file_ref(path),
        "source": copy.deepcopy(build_source),
        "binary": binary,
        "command": {"argv": argv, "returncode": 0},
        "http": None,
        "product": {"success": True, "command_rc": 0},
        "target_signal": target_signal,
        "harness": {"status": "pass", "failure_class": None},
        "witness_hardware": witness_hardware,
        "model": expected_model_identity,
        "request_bundle": copy.deepcopy(bundles[0]),
        "artifacts": artifacts,
    }


def validate_raw_execution(
    path: Path,
    *,
    check_id: str,
    entrypoint: str,
    profile_detail: str,
    build_source: dict[str, Any],
    binary_sha256: str,
    require_timing_metrics: bool = False,
    expected_build_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = read_json(path, f"{check_id} raw execution")
    if raw.get("schema_version") == 2:
        require(
            require_timing_metrics,
            f"{check_id} schema v2 is reserved for formal profile timing",
        )
        return _validate_raw_execution_v2(
            path,
            check_id=check_id,
            entrypoint=entrypoint,
            profile_detail=profile_detail,
            build_source=build_source,
            binary_sha256=binary_sha256,
            expected_build_receipt=expected_build_receipt,
        )
    require(
        not require_timing_metrics,
        f"{check_id} formal profile timing requires raw execution schema v2",
    )
    return _validate_raw_execution_v1(
        path,
        check_id=check_id,
        entrypoint=entrypoint,
        profile_detail=profile_detail,
        build_source=build_source,
        binary_sha256=binary_sha256,
        require_timing_metrics=False,
        expected_build_receipt=expected_build_receipt,
    )


def validate_witness(
    path: Path,
    *,
    check_id: str,
    entrypoint: str,
    profile_detail: str,
    build_source: dict[str, Any],
    binary_sha256: str,
    require_timing_metrics: bool = False,
    expected_build_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    witness = read_json(path, f"{check_id} witness")
    required_fields = {
        "schema_version",
        "artifact_type",
        "status",
        "check_id",
        "source_git_sha",
        "source_tree_sha",
        "backend",
        "entrypoint",
        "profile_detail",
        "binary_sha256",
        "product",
        "target_signal",
        "harness",
        "evidence",
    }
    require(set(witness) == required_fields, f"{check_id} witness field set differs")
    require(
        witness.get("schema_version") == 1
        and witness.get("artifact_type") == WITNESS_TYPE
        and witness.get("status") == "target_signal_pass"
        and witness.get("check_id") == check_id
        and witness.get("source_git_sha") == build_source["git_sha"]
        and witness.get("source_tree_sha") == build_source["git_tree_sha"]
        and witness.get("backend") == "cuda"
        and witness.get("entrypoint") == entrypoint
        and witness.get("profile_detail") == profile_detail
        and witness.get("binary_sha256") == binary_sha256,
        f"{check_id} witness identity differs",
    )
    evidence = witness.get("evidence")
    require(
        isinstance(evidence, dict) and set(evidence) == {"raw_execution"},
        f"{check_id} evidence must name one raw execution receipt",
    )
    raw_path = validate_ref(
        evidence.get("raw_execution"), f"{check_id} raw execution"
    )
    derived = validate_raw_execution(
        raw_path,
        check_id=check_id,
        entrypoint=entrypoint,
        profile_detail=profile_detail,
        build_source=build_source,
        binary_sha256=binary_sha256,
        require_timing_metrics=require_timing_metrics,
        expected_build_receipt=expected_build_receipt,
    )
    require(
        witness.get("product") == derived["product"]
        and derived["product"].get("success") is True,
        f"{check_id} product summary differs from raw evidence",
    )
    require(
        witness.get("target_signal") == derived["target_signal"]
        and derived["target_signal"]["startup_catalog_error_count"] == 0
        and (
            (
                profile_detail == "full"
                and derived["target_signal"][
                    "startup_reusable_execution_span_count"
                ]
                > 0
                and derived["target_signal"][
                    "startup_reusable_program_fingerprint_count"
                ]
                > 0
            )
            or (
                profile_detail == "verify"
                and derived["target_signal"][
                    "startup_verification_request_completed_count"
                ]
                > 0
            )
        ),
        f"{check_id} startup target signal differs from raw evidence",
    )
    require(
        witness.get("harness") == derived["harness"],
        f"{check_id} harness summary differs from raw evidence",
    )
    if entrypoint == "run":
        require(
            derived["harness"] == {"status": "pass", "failure_class": None},
            "run/full harness did not pass",
        )
    result = {
        "check_id": check_id,
        "witness": file_ref(path),
        "raw_execution": derived,
    }
    if require_timing_metrics:
        result["witness_hardware"] = copy.deepcopy(derived["witness_hardware"])
    return result


def backend_partition(base_source: dict[str, Any], head_source: dict[str, Any]) -> dict[str, Any]:
    backend_root = "crates/ferrum-kernels/src/backend"
    expected = ["crates/ferrum-kernels/src/backend/cuda/vnext_ops.rs"]
    rows: dict[str, list[str]] = {}
    for label, source in (("base", base_source), ("head", head_source)):
        process = git_process(
            [
                "grep",
                "-l",
                "DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID",
                source["git_sha"],
                "--",
                backend_root,
            ],
            text=True,
        )
        require(process.returncode in {0, 1}, f"backend capability grep failed for {label}")
        paths = sorted(
            line.split(":", 1)[1] if ":" in line else line
            for line in process.stdout.splitlines()
            if line
        )
        require(paths == expected, f"reusable capability backend partition differs at {label}: {paths}")
        rows[label] = paths
    return {
        "capability_id": "DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID",
        "base_registrations": rows["base"],
        "head_registrations": rows["head"],
        "affected_backends": ["cuda"],
        "unaffected_backends": ["metal"],
    }


def source_file_versions(
    base_source: dict[str, Any], source: dict[str, Any], changed: list[str]
) -> dict[str, dict[str, str]]:
    versions: dict[str, dict[str, str]] = {}
    for path in changed:
        before = git_process(["show", f"{base_source['git_sha']}:{path}"])
        after = git_process(["show", f"{source['git_sha']}:{path}"])
        if before.returncode == 0 and after.returncode == 0:
            versions[path] = {
                "before": bytes(before.stdout).decode("utf-8"),
                "after": bytes(after.stdout).decode("utf-8"),
            }
    return versions


def validate_profile_timing_plan_closure(
    *,
    changed: list[str],
    plan: dict[str, Any],
    matches: list[dict[str, Any]],
    rule_config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    rules = rule_config.get("rules")
    require(isinstance(rules, list), "change-impact rule set is missing")
    control_rules = [
        rule
        for rule in rules
        if isinstance(rule, dict) and rule.get("id") == CONTROL_PLANE_RULE_ID
    ]
    require(
        len(control_rules) == 1,
        "change-impact control-plane rule identity differs",
    )
    control_rule = control_rules[0]
    require(
        control_rule.get("exclusive") is True
        and sorted(control_rule.get("path_globs", [])) == list(CONTROL_PLANE_PATHS)
        and control_rule.get("domains") == CONTROL_PLANE_DOMAINS
        and sorted(control_rule.get("required_gates", []))
        == CONTROL_PLANE_REQUIRED_GATES
        and sorted(control_rule.get("release_invalidation", []))
        == CONTROL_PLANE_REQUIRED_GATES
        and control_rule.get("required_scenarios", []) == []
        and control_rule.get("qualification_profiles", []) == []
        and control_rule.get("exceptions", []) == [],
        "change-impact control-plane rule contract differs",
    )
    allowed_control = set(CONTROL_PLANE_PATHS)
    control_changed = sorted(path for path in changed if path in allowed_control)
    qualified_changed = sorted(path for path in changed if path not in allowed_control)
    matched_paths = sorted(str(match.get("path")) for match in matches)
    require(
        len(matched_paths) == len(set(matched_paths))
        and matched_paths == qualified_changed,
        "profile timing qualification did not select every non-control changed path",
    )
    require(
        set(path for path in changed if is_release_product_path(path))
        <= set(matched_paths),
        "profile timing qualification left a product path unqualified",
    )
    decision_log = plan.get("decision_log")
    require(isinstance(decision_log, list), "change-impact decision log is missing")
    control_decisions = [
        row
        for row in decision_log
        if isinstance(row, dict) and row.get("rule_id") == CONTROL_PLANE_RULE_ID
    ]
    require(
        sorted(str(row.get("path")) for row in control_decisions)
        == control_changed,
        "control-plane path classification differs",
    )
    for row in control_decisions:
        require(
            row.get("path") in allowed_control
            and row.get("domains") == CONTROL_PLANE_DOMAINS
            and row.get("required_product_scenarios_added") == []
            and row.get("release_invalidation")
            == CONTROL_PLANE_REQUIRED_GATES,
            "control-plane decision escaped its exact domain/gate contract",
        )
    expected_domains = ["diagnostic_observability"]
    if control_changed:
        expected_domains = sorted(
            [*CONTROL_PLANE_DOMAINS, "diagnostic_observability"]
        )
    require(
        plan.get("impact_domains") == expected_domains,
        "profile timing qualification impact domain is not exactly closed",
    )
    require(
        plan.get("required_gates") == EXPECTED_CONTROL_GATES,
        "profile timing control gates are not exactly closed",
    )
    require(
        plan.get("required_product_scenarios") == [],
        "profile timing qualification unexpectedly selected product scenarios",
    )
    return qualified_changed, control_changed


def plan_classification(
    base_source: dict[str, Any],
    source: dict[str, Any],
    rules_path: Path,
    planner: ModuleType,
    profile_id: str,
) -> dict[str, Any]:
    changed = changed_paths(base_source["git_sha"], source["git_sha"])
    rule_config = planner.load_rule_config(rules_path)
    plan = planner.plan_from_files(
        changed_files=changed,
        base_sha=base_source["git_sha"],
        head_sha=source["git_sha"],
        dirty=False,
        rules=rule_config["rules"],
        qualification_profiles=rule_config["qualification_profiles"],
        file_versions=source_file_versions(base_source, source, changed),
    )
    matches = [
        match
        for match in plan["qualification_matches"]
        if match.get("profile_id") == profile_id
    ]
    expected_checks, expected_scopes = qualification_profile_contract(profile_id)
    product_changed = [path for path in changed if is_release_product_path(path)]
    require(plan.get("status") == "pass" and not plan.get("unknown_files"), "change-impact plan did not close")
    require(matches, "qualification profile did not match the product diff")
    require(
        plan.get("required_checks") == expected_checks,
        "qualification required check set differs",
    )
    require(
        plan.get("qualified_scopes") == expected_scopes,
        "qualification scope set differs",
    )
    expected_control_gates = (
        LEGACY_EXPECTED_CONTROL_GATES
        if profile_id == LEGACY_PROFILE_ID
        else EXPECTED_CONTROL_GATES
    )
    require(
        plan.get("required_gates") == expected_control_gates,
        "qualification control gate set is not exactly closed",
    )
    control_changed = [path for path in changed if path not in product_changed]
    if profile_id == PROFILE_ID:
        _, control_changed = validate_profile_timing_plan_closure(
            changed=changed,
            plan=plan,
            matches=matches,
            rule_config=rule_config,
        )
    selector: dict[str, Any] = (
        matches[0]
        if len(matches) == 1
        else {"matches": matches}
    )
    causal_edges = (
        [
            {
                "from": "executor.startup_reusable_program_identity",
                "to": "cuda.run.profile_full",
                "reason": "CUDA reusable capability plus Kernel timing startup",
            },
            {
                "from": "executor.startup_reusable_program_identity",
                "to": "cuda.serve.profile_verify",
                "reason": "CUDA reusable capability plus Verification timing startup",
            },
        ]
        if profile_id == LEGACY_PROFILE_ID
        else [
            {
                "from": f"qualification.{profile_id}",
                "to": (
                    f"{scope['backend']}.{scope['entrypoint']}."
                    f"profile_{scope['profile_detail']}"
                ),
                "reason": "machine-selected diagnostic observability consumer",
            }
            for scope in expected_scopes
        ]
    )
    return {
        "profile_id": profile_id,
        "selector": selector,
        "required_checks": expected_checks,
        "qualified_scopes": expected_scopes,
        "product_changed_files": product_changed,
        "control_plane_changed_files": control_changed,
        "impact_domains": plan["impact_domains"],
        "required_control_gates": plan["required_gates"],
        "causal_edges": causal_edges,
    }


def qualification_profile_contract(
    profile_id: str,
) -> tuple[list[str], list[dict[str, str]]]:
    if profile_id == PROFILE_ID:
        return copy.deepcopy(EXPECTED_CHECKS), copy.deepcopy(EXPECTED_SCOPES)
    if profile_id == LEGACY_PROFILE_ID:
        return copy.deepcopy(LEGACY_EXPECTED_CHECKS), copy.deepcopy(
            LEGACY_EXPECTED_SCOPES
        )
    raise QualificationError(f"unsupported qualification profile: {profile_id}")


def plan_classification_at_source(
    base_source: dict[str, Any], source: dict[str, Any], profile_id: str
) -> dict[str, Any]:
    temporary, planner, rules_path = historical_planner(source)
    try:
        return plan_classification(
            base_source, source, rules_path, planner, profile_id
        )
    finally:
        temporary.cleanup()


def docs_review_closure(
    classification: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    docs = sorted(
        path
        for path in classification["control_plane_changed_files"]
        if path.startswith("docs/")
    )
    goal_root = "docs/goals/runtime-vnext-0.8.0-2026-07-10/"
    if not docs:
        docs = [
            goal_root + "CHANGE_IMPACT_REGRESSION_PLAN_2026-08-12.md"
        ]
    require(
        all(path.startswith(goal_root) for path in docs),
        f"qualification docs review escaped the active goal: {docs}",
    )
    return {
        "gate": "docs_review",
        "status": "closed",
        "method": "active_goal_policy_and_acceptance_review",
        "documents": [git_blob_ref(source, path) for path in docs],
    }


def revalidated_cells(
    binary_sha256: str, profile_id: str = PROFILE_ID
) -> list[dict[str, Any]]:
    _, scopes = qualification_profile_contract(profile_id)
    return [
        {
            "cell_id": (
                f"{scope['backend']}.{scope['entrypoint']}."
                f"profile_{scope['profile_detail']}"
            ),
            "backend": scope["backend"],
            "entrypoint": scope["entrypoint"],
            "profile_detail": scope["profile_detail"],
            "evidence": "diagnostic_observability",
            "check_id": (
                f"{scope['backend']}_{scope['entrypoint']}_"
                f"profile_{scope['profile_detail']}"
            ),
            "binary_sha256": binary_sha256,
        }
        for scope in scopes
    ]


def diff_identity(base_source: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    raw = git_bytes(
        "diff",
        "--raw",
        "-z",
        "--no-abbrev",
        f"{base_source['git_sha']}..{source['git_sha']}",
        "--",
    )
    patch = git_bytes(
        "diff",
        "--binary",
        "--full-index",
        f"{base_source['git_sha']}..{source['git_sha']}",
        "--",
    )
    return {
        "changed_files": changed_paths(
            base_source["git_sha"], source["git_sha"]
        ),
        "raw_diff_sha256": sha256_bytes(raw),
        "raw_diff_size_bytes": len(raw),
        "binary_patch_sha256": sha256_bytes(patch),
        "binary_patch_size_bytes": len(patch),
    }


def qualification_document(
    *,
    output: Path,
    source: dict[str, Any],
    prior_path: Path,
    prior: dict[str, Any],
    candidate: dict[str, Any],
    exact_contracts: dict[str, Any],
    collector_selftest: dict[str, Any],
    control_gate: dict[str, Any],
    run_full: dict[str, Any],
) -> dict[str, Any]:
    base_source = prior["source"]
    classification = plan_classification_at_source(base_source, source, PROFILE_ID)
    affected_backends = sorted(
        {scope["backend"] for scope in classification["qualified_scopes"]}
    )
    partition = {
        "source": "planner_qualified_scopes",
        "affected_backends": affected_backends,
        "unaffected_backends": sorted(set(("cuda", "metal")) - set(affected_backends)),
    }
    authority = {
        "cuda": candidate["binary_sha256"],
        "metal": prior["backend_binary_sha256"]["metal"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "pass",
        "profile_id": PROFILE_ID,
        "artifact_dir": str(output),
        "source": source,
        "prior_source": base_source,
        "prior_r1": file_ref(prior_path),
        "canonical_inputs": canonical_input_refs(source),
        "diff": diff_identity(base_source, source),
        "classification": classification,
        "proofs": {
            "candidate_cuda": candidate,
            "profile_timing_exact_contracts": exact_contracts,
            "profile_collector_selftest": collector_selftest,
            "release_validator_selftest": control_gate,
            "docs_review": docs_review_closure(classification, source),
            "cuda_run_profile_full": run_full,
            "backend_scope_partition": partition,
        },
        "reused_cells": prior["reused_cells"],
        "revalidated_cells": revalidated_cells(candidate["binary_sha256"], PROFILE_ID),
        "invalidated_cells": [],
        "open_invalidated_cells": [],
        "backend_binary_sha256": authority,
        "prior_reachability": prior["reachability"],
        "witness_hardware": copy.deepcopy(run_full["witness_hardware"]),
        "does_not_prove": DOES_NOT_PROVE,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "pass_line": f"{PASS_PREFIX}: {output}",
    }


def build(
    *,
    prior_r1: Path,
    profile_id: str,
    candidate_cuda_binary: Path,
    candidate_cuda_build_receipt: Path,
    exact_contracts_receipt: Path,
    profile_collector_selftest_receipt: Path,
    control_gate_receipt: Path,
    run_full_witness: Path,
    out: Path,
) -> str:
    require(profile_id == PROFILE_ID, f"unsupported qualification profile: {profile_id}")
    output = out.expanduser().resolve()
    require(
        output != REPO_ROOT and REPO_ROOT not in output.parents,
        "qualification output must be outside the source tree",
    )
    require(
        not output.exists() or not any(output.iterdir()),
        f"qualification output must be absent or empty: {output}",
    )
    source = current_source()
    prior_path = prior_r1.expanduser().resolve()
    prior = prior_r1_summary(prior_path)
    candidate = validate_candidate_build(
        candidate_cuda_binary.expanduser().resolve(),
        candidate_cuda_build_receipt.expanduser().resolve(),
        source,
        require_native_op_artifact=True,
    )
    exact_contracts = validate_profile_timing_exact_contracts_receipt(
        exact_contracts_receipt.expanduser().resolve(), source
    )
    collector_selftest = validate_profile_collector_selftest_receipt(
        profile_collector_selftest_receipt.expanduser().resolve(), source
    )
    control_gate = validate_control_gate_receipt(
        control_gate_receipt.expanduser().resolve(), source
    )
    run_full = validate_witness(
        run_full_witness.expanduser().resolve(),
        check_id="cuda_run_profile_full",
        entrypoint="run",
        profile_detail="full",
        build_source=candidate["build_source"],
        binary_sha256=candidate["binary_sha256"],
        require_timing_metrics=True,
        expected_build_receipt=candidate["build_receipt"],
    )
    require(current_source() == source, "qualification source changed during validation")
    output.mkdir(parents=True, exist_ok=True)
    try:
        document = qualification_document(
            output=output,
            source=source,
            prior_path=prior_path,
            prior=prior,
            candidate=candidate,
            exact_contracts=exact_contracts,
            collector_selftest=collector_selftest,
            control_gate=control_gate,
            run_full=run_full,
        )
        write_json(output / "manifest.json", document)
        verify_manifest(output / "manifest.json", verify_checkout=True)
        return str(document["pass_line"])
    except BaseException:
        if output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        raise


def validate_qualification_envelope(
    path: Path,
    manifest: dict[str, Any],
    *,
    verify_checkout: bool,
    expected_source: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    root = path.parent
    profile_id = manifest.get("profile_id")
    require(isinstance(profile_id, str), "qualification profile ID is invalid")
    contract_id = QUALIFICATION_CONTRACTS.get(
        (manifest.get("schema_version"), manifest.get("artifact_type"), profile_id)
    )
    require(contract_id is not None, "qualification validator contract is unsupported")
    expected_fields = (
        PROFILE_TIMING_MANIFEST_FIELDS
        if contract_id == "profile-timing-v1"
        else QUALIFICATION_MANIFEST_FIELDS
    )
    require(
        set(manifest) == expected_fields,
        "qualification manifest field set differs",
    )
    qualification_profile_contract(profile_id)
    require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type") == ARTIFACT_TYPE
        and manifest.get("status") == "pass"
        and Path(str(manifest.get("artifact_dir", ""))).resolve() == root
        and manifest.get("does_not_prove") == DOES_NOT_PROVE
        and manifest.get("pass_line") == f"{PASS_PREFIX}: {root}",
        "qualification identity/status/PASS differs",
    )
    source = normalize_source(manifest.get("source"), "qualification")
    expected = current_source() if verify_checkout else expected_source
    if expected is not None:
        require(source == expected, "qualification source is stale")
    return source, contract_id


def _verify_manifest_uncached(
    manifest_path: Path,
    *,
    verify_checkout: bool = True,
    expected_source: dict[str, Any] | None = None,
    _verification_context: Any | None = None,
) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve()
    manifest = read_json(path, "change-impact qualification manifest")
    source, contract_id = validate_qualification_envelope(
        path,
        manifest,
        verify_checkout=verify_checkout,
        expected_source=expected_source,
    )
    profile_id = manifest["profile_id"]
    prior_path = validate_ref(manifest.get("prior_r1"), "prior R1 manifest")
    prior = prior_r1_summary(
        prior_path, _verification_context=_verification_context
    )
    require(
        manifest.get("prior_source") == prior["source"],
        "qualification prior source differs",
    )
    canonical = manifest.get("canonical_inputs")
    require(isinstance(canonical, dict), "qualification canonical inputs are missing")
    validate_canonical_input_refs(canonical, source)
    require(
        manifest.get("diff") == diff_identity(prior["source"], source),
        "qualification diff identity differs",
    )
    classification = plan_classification_at_source(
        prior["source"], source, profile_id
    )
    require(
        manifest.get("classification") == classification,
        "qualification classification differs",
    )
    proofs = manifest.get("proofs")
    legacy = contract_id == "legacy-startup-v1"
    expected_proofs = (
        {
            "candidate_cuda",
            "unit_exact",
            "release_validator_selftest",
            "docs_review",
            "cuda_run_profile_full",
            "cuda_serve_profile_verify",
            "backend_capability_partition",
        }
        if legacy
        else {
            "candidate_cuda",
            "profile_timing_exact_contracts",
            "profile_collector_selftest",
            "release_validator_selftest",
            "docs_review",
            "cuda_run_profile_full",
            "backend_scope_partition",
        }
    )
    require(
        isinstance(proofs, dict) and set(proofs) == expected_proofs,
        "qualification proof set differs",
    )
    candidate_recorded = proofs["candidate_cuda"]
    require(isinstance(candidate_recorded, dict), "candidate CUDA proof is invalid")
    binary_path = validate_ref(candidate_recorded.get("binary"), "candidate CUDA binary")
    receipt_path = validate_ref(
        candidate_recorded.get("build_receipt"), "candidate CUDA build receipt"
    )
    candidate = validate_candidate_build(
        binary_path,
        receipt_path,
        source,
        require_native_op_artifact=not legacy,
    )
    require(candidate_recorded == candidate, "candidate CUDA proof drifted")
    if legacy:
        unit_recorded = proofs["unit_exact"]
        require(isinstance(unit_recorded, dict), "unit proof is invalid")
        unit_path = validate_ref(unit_recorded.get("receipt"), "exact unit receipt")
        require(
            unit_recorded == validate_unit_receipt(unit_path, source),
            "unit proof drifted",
        )
    else:
        exact_recorded = proofs["profile_timing_exact_contracts"]
        collector_recorded = proofs["profile_collector_selftest"]
        require(
            isinstance(exact_recorded, dict)
            and isinstance(collector_recorded, dict),
            "profile timing focused proof is invalid",
        )
        exact_path = validate_ref(
            exact_recorded.get("receipt"), "profile timing exact contracts receipt"
        )
        collector_path = validate_ref(
            collector_recorded.get("receipt"), "profile collector self-test receipt"
        )
        require(
            exact_recorded
            == validate_profile_timing_exact_contracts_receipt(exact_path, source)
            and collector_recorded
            == validate_profile_collector_selftest_receipt(collector_path, source),
            "profile timing focused proof drifted",
        )
    control_recorded = proofs["release_validator_selftest"]
    require(isinstance(control_recorded, dict), "control gate proof is invalid")
    control_path = validate_ref(
        control_recorded.get("receipt"), "qualification control self-test receipt"
    )
    require(
        control_recorded == validate_control_gate_receipt(control_path, source),
        "qualification control gate proof drifted",
    )
    docs_recorded = proofs["docs_review"]
    require(
        docs_recorded == docs_review_closure(classification, source),
        "qualification docs review closure drifted",
    )
    proved_control_gates = sorted(
        [docs_recorded["gate"], *control_recorded["gates"]]
    )
    require(
        classification["required_control_gates"] == proved_control_gates,
        "qualification control gates are not fully proved",
    )
    run_recorded = proofs["cuda_run_profile_full"]
    require(isinstance(run_recorded, dict), "run witness proof is invalid")
    run_path = validate_ref(run_recorded.get("witness"), "CUDA run/full witness")
    run_full = validate_witness(
        run_path,
        check_id="cuda_run_profile_full",
        entrypoint="run",
        profile_detail="full",
        build_source=candidate["build_source"],
        binary_sha256=candidate["binary_sha256"],
        require_timing_metrics=not legacy,
        expected_build_receipt=(None if legacy else candidate["build_receipt"]),
    )
    require(run_recorded == run_full, "CUDA run/full witness proof drifted")
    if legacy:
        serve_recorded = proofs["cuda_serve_profile_verify"]
        require(isinstance(serve_recorded, dict), "serve witness proof is invalid")
        serve_path = validate_ref(serve_recorded.get("witness"), "CUDA serve/verify witness")
        serve_verify = validate_witness(
            serve_path,
            check_id="cuda_serve_profile_verify",
            entrypoint="serve",
            profile_detail="verify",
            build_source=candidate["build_source"],
            binary_sha256=candidate["binary_sha256"],
        )
        require(serve_recorded == serve_verify, "CUDA serve/verify witness proof drifted")
        partition = backend_partition(prior["source"], source)
        require(
            proofs["backend_capability_partition"] == partition,
            "backend capability partition drifted",
        )
    else:
        affected_backends = sorted(
            {scope["backend"] for scope in classification["qualified_scopes"]}
        )
        partition = {
            "source": "planner_qualified_scopes",
            "affected_backends": affected_backends,
            "unaffected_backends": sorted(
                set(("cuda", "metal")) - set(affected_backends)
            ),
        }
        require(
            proofs["backend_scope_partition"] == partition,
            "backend scope partition drifted",
        )
    reused = prior["reused_cells"]
    revalidated = revalidated_cells(candidate["binary_sha256"], profile_id)
    require(manifest.get("reused_cells") == reused, "R1 reused cell denominator drifted")
    require(
        manifest.get("revalidated_cells") == revalidated,
        "revalidated cell set drifted",
    )
    require(
        manifest.get("invalidated_cells") == []
        and manifest.get("open_invalidated_cells") == [],
        "qualification still has open affected cells",
    )
    authority = {
        "cuda": candidate["binary_sha256"],
        "metal": prior["backend_binary_sha256"]["metal"],
    }
    require(
        manifest.get("backend_binary_sha256") == authority,
        "qualified backend binary authority differs",
    )
    require(
        manifest.get("prior_reachability") == prior["reachability"],
        "prior R1 reachability proof drifted",
    )
    if not legacy:
        require(
            manifest.get("witness_hardware") == run_full["witness_hardware"],
            "qualification witness hardware drifted",
        )
    summary = {
        "kind": "runtime-vnext-change-impact-qualification",
        "manifest": file_ref(path),
        "profile_id": profile_id,
        "source": source,
        "prior_source": prior["source"],
        "prior_r1": file_ref(prior_path),
        "selector": classification["selector"],
        "qualified_scopes": classification["qualified_scopes"],
        "reused_cells": copy.deepcopy(reused),
        "revalidated_cells": copy.deepcopy(revalidated),
        "invalidated_cells": [],
        "open_invalidated_cells": [],
        "backend_binary_sha256": authority,
        "proofs": copy.deepcopy(proofs),
    }
    if not legacy:
        summary["witness_hardware"] = copy.deepcopy(run_full["witness_hardware"])
    return summary


def verify_manifest(
    manifest_path: Path,
    *,
    verify_checkout: bool = True,
    expected_source: dict[str, Any] | None = None,
    _verification_context: Any | None = None,
) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve()
    manifest = read_json(path, "change-impact qualification manifest")
    validate_qualification_envelope(
        path,
        manifest,
        verify_checkout=verify_checkout,
        expected_source=expected_source,
    )
    try:
        import runtime_vnext_r1_product_correctness as r1_correctness

        context = r1_correctness._coerce_verification_context(
            _verification_context
        )
    except (ImportError, RuntimeError, TypeError, ValueError) as error:
        raise QualificationError(f"R1 verification context failed: {error}") from error
    artifact_key = r1_correctness._artifact_key("qualification", path)
    require(
        artifact_key not in context.active,
        "R1/qualification dependency cycle detected",
    )
    cached = context.qualification_memo.get(artifact_key)
    if cached is not None:
        return copy.deepcopy(cached)
    context.active.add(artifact_key)
    try:
        summary = _verify_manifest_uncached(
            path,
            verify_checkout=verify_checkout,
            expected_source=expected_source,
            _verification_context=context,
        )
        context.qualification_memo[artifact_key] = copy.deepcopy(summary)
        return copy.deepcopy(summary)
    finally:
        context.active.remove(artifact_key)


def expect_reject(action: Any, marker: str) -> None:
    try:
        action()
    except (QualificationError, OSError, ValueError):
        return
    raise QualificationError(f"self-test mutation was accepted: {marker}")


def self_test() -> int:
    acceptance = {
        "total_matrix_case_count": 1867,
        "models": {
            "m1_cuda": {"backend": "cuda", "cases": "703/703"},
            "m1_metal": {"backend": "metal", "cases": "702/702"},
            "m2_cuda": {"backend": "cuda", "cases": "112/112"},
            "m2_metal": {"backend": "metal", "cases": "111/111"},
            "m3_cuda": {"backend": "cuda", "cases": "120/120"},
            "m3_metal": {"backend": "metal", "cases": "119/119"},
        },
    }
    reused = derive_reused_cells(acceptance)
    require(
        len(reused) == 8
        and sum(row.get("case_count", 0) for row in reused) == 1867
        and sum(row.get("scenario_count", 0) for row in reused) == 6,
        "self-test reuse denominator differs",
    )
    binary = "1" * 64
    cells = revalidated_cells(binary)
    require(
        [row["cell_id"] for row in cells] == ["cuda.run.profile_full"]
        and all(row["binary_sha256"] == binary for row in cells),
        "self-test revalidated cell set differs",
    )
    require(
        [row["cell_id"] for row in revalidated_cells(binary, LEGACY_PROFILE_ID)]
        == ["cuda.run.profile_full", "cuda.serve.profile_verify"],
        "self-test legacy revalidated cell set differs",
    )
    expect_reject(
        lambda: derive_reused_cells(
            {**acceptance, "total_matrix_case_count": 1866}
        ),
        "wrong R1 denominator",
    )
    require_sha256(binary, "self-test binary")
    expect_reject(lambda: require_sha256("bad", "self-test binary"), "bad SHA")
    rule_config = plan_gates.load_rule_config(RULES_PATH)
    profile = rule_config["qualification_profiles"].get(PROFILE_ID)
    require(
        isinstance(profile, dict)
        and sorted(profile.get("required_checks", [])) == EXPECTED_CHECKS
        and profile.get("qualified_scopes") == EXPECTED_SCOPES,
        "self-test canonical qualification profile differs",
    )
    mixed_product_path = "crates/ferrum-engine/src/continuous_engine/inner/decode.rs"
    mixed_changed = sorted([mixed_product_path, *CONTROL_PLANE_PATHS])
    mixed_matches = [{"profile_id": PROFILE_ID, "path": mixed_product_path}]
    mixed_plan = {
        "impact_domains": sorted(
            [*CONTROL_PLANE_DOMAINS, "diagnostic_observability"]
        ),
        "required_gates": EXPECTED_CONTROL_GATES,
        "required_product_scenarios": [],
        "decision_log": [
            {
                "path": path,
                "rule_id": CONTROL_PLANE_RULE_ID,
                "domains": CONTROL_PLANE_DOMAINS,
                "required_product_scenarios_added": [],
                "release_invalidation": CONTROL_PLANE_REQUIRED_GATES,
            }
            for path in CONTROL_PLANE_PATHS
        ],
    }
    mixed_qualified, mixed_control = validate_profile_timing_plan_closure(
        changed=mixed_changed,
        plan=mixed_plan,
        matches=mixed_matches,
        rule_config=rule_config,
    )
    require(
        mixed_qualified == [mixed_product_path]
        and mixed_control == list(CONTROL_PLANE_PATHS),
        "mixed product/control qualification closure differs",
    )
    forged_control_path = "scripts/release/runtime_vnext_forged_control.py"
    forged_control_plan = copy.deepcopy(mixed_plan)
    forged_control_plan["decision_log"].append(
        {
            "path": forged_control_path,
            "rule_id": CONTROL_PLANE_RULE_ID,
            "domains": CONTROL_PLANE_DOMAINS,
            "required_product_scenarios_added": [],
            "release_invalidation": CONTROL_PLANE_REQUIRED_GATES,
        }
    )
    expect_reject(
        lambda: validate_profile_timing_plan_closure(
            changed=sorted([*mixed_changed, forged_control_path]),
            plan=forged_control_plan,
            matches=mixed_matches,
            rule_config=rule_config,
        ),
        "forged path classified as qualification control plane",
    )
    broadened_rules = copy.deepcopy(rule_config)
    control_rule = next(
        rule
        for rule in broadened_rules["rules"]
        if rule.get("id") == CONTROL_PLANE_RULE_ID
    )
    control_rule["path_globs"].append(forged_control_path)
    expect_reject(
        lambda: validate_profile_timing_plan_closure(
            changed=mixed_changed,
            plan=mixed_plan,
            matches=mixed_matches,
            rule_config=broadened_rules,
        ),
        "self-broadened control-plane allowlist",
    )
    import runtime_vnext_r2_profile_collector as profile_collector

    cuda_profile_fixture = profile_collector.fixture_profile_events("cuda")[2]
    timing_metrics = derive_cuda_full_profile_metrics(
        cuda_profile_fixture, "self-test CUDA run/full"
    )
    require(
        timing_metrics["stage_coverage"] >= MIN_STAGE_COVERAGE
        and timing_metrics["clock_relative_error_fraction"]
        <= MAX_CLOCK_RELATIVE_ERROR
        and timing_metrics["dispatch_timing_coverage"]
        >= MIN_DISPATCH_TIMING_COVERAGE
        and timing_metrics["device_attribution_coverage"]
        >= MIN_DEVICE_ATTRIBUTION_COVERAGE,
        "self-test CUDA timing metrics differ",
    )
    incomplete_interval_fixture = copy.deepcopy(cuda_profile_fixture)
    incomplete_native = next(
        event
        for event in incomplete_interval_fixture
        if event.get("phase") == "vnext.device_native_work"
    )
    incomplete_native["backend_detail"]["device_intervals"][0][
        "end_offset_ns"
    ] = 101
    expect_reject(
        lambda: derive_cuda_full_profile_metrics(
            incomplete_interval_fixture, "self-test incomplete device interval"
        ),
        "CUDA timing metrics with incomplete device interval attribution",
    )
    overlapping_interval_fixture = copy.deepcopy(cuda_profile_fixture)
    overlapping_native = next(
        event
        for event in overlapping_interval_fixture
        if event.get("phase") == "vnext.device_native_work"
    )
    overlapping_native["backend_detail"]["device_intervals"] = [
        {
            "start_offset_ns": 100,
            "end_offset_ns": 550,
            "subwork_id": "kernel.decode.first",
        },
        {
            "start_offset_ns": 500,
            "end_offset_ns": 950,
            "subwork_id": "kernel.decode.second",
        },
    ]
    expect_reject(
        lambda: derive_cuda_full_profile_metrics(
            overlapping_interval_fixture, "self-test overlapping device intervals"
        ),
        "CUDA timing metrics with overlapping device intervals",
    )
    missing_stage_fixture = copy.deepcopy(cuda_profile_fixture)
    terminal = next(
        event
        for event in missing_stage_fixture
        if event.get("phase") == "actual_run_generation"
    )
    terminal["attributes"].pop("engine_decode_stage_intervals")
    expect_reject(
        lambda: derive_cuda_full_profile_metrics(
            missing_stage_fixture, "self-test missing stage"
        ),
        "CUDA timing metrics without engine stages",
    )
    with tempfile.TemporaryDirectory(
        prefix="ferrum-vnext-impact-witness-selftest-"
    ) as temp_name:
        temp_root = Path(temp_name)
        binary_path = temp_root / "ferrum"
        binary_path.write_bytes(b"self-test CUDA binary")
        raw_binary_sha = sha256(binary_path)
        build_source = source_at(git_text("rev-parse", "HEAD"))
        build_command = [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--features",
            "cuda,vllm-moe-marlin,vllm-paged-attn-v2,native-op-artifact",
        ]
        bounded_path = temp_root / "build.bounded.receipt.json"
        write_json(
            bounded_path,
            {
                "schema": "ferrum.bounded-command-receipt.v1",
                "status": "pass",
                "rc": 0,
                "reason": "command_completed",
                "command": build_command,
                "violation": None,
                "cleanup": {"process_group_gone": True},
                "termination": {"errors": [], "signals": []},
                "sampling_error_count": 0,
                "sampling_errors": [],
            },
        )
        native_artifact = temp_root / "native-operator.bin"
        native_evidence = temp_root / "native-operator.receipt.json"
        native_artifact.write_bytes(b"native operator fixture")
        write_json(native_evidence, {"status": "pass"})
        native_lock_path = temp_root / "native-operators.lock.json"
        write_json(
            native_lock_path,
            {
                "artifacts": [
                    {
                        "artifact_path": native_artifact.name,
                        "binary_sha256": sha256(native_artifact),
                        "evidence": {
                            "path": native_evidence.name,
                            "sha256": sha256(native_evidence),
                            "size_bytes": native_evidence.stat().st_size,
                        },
                    }
                ]
            },
        )
        build_receipt_path = temp_root / "candidate-build-receipt.json"
        write_json(
            build_receipt_path,
            {
                "schema_version": 1,
                "artifact_type": "runtime_vnext_candidate_build_receipt",
                "status": "pass",
                "backend": "cuda",
                "returncode": 0,
                "binary_sha256": raw_binary_sha,
                "source_git_sha": build_source["git_sha"],
                "source_tree_sha": build_source["git_tree_sha"],
                "dirty_status": {"is_dirty": False, "status_short": []},
                "command": build_command,
                "bounded_receipt": file_ref(bounded_path),
                "native_operator_set_lock": file_ref(native_lock_path),
            },
        )
        candidate_fixture = validate_candidate_build(
            binary_path,
            build_receipt_path,
            build_source,
            require_native_op_artifact=True,
        )
        require(
            candidate_fixture["binary_sha256"] == raw_binary_sha
            and candidate_fixture["native_operator_closure_sha256"],
            "self-test bounded native candidate build differs",
        )
        require(
            QUALIFICATION_CONTRACTS
            == {
                (SCHEMA_VERSION, ARTIFACT_TYPE, PROFILE_ID): "profile-timing-v1",
                (SCHEMA_VERSION, ARTIFACT_TYPE, LEGACY_PROFILE_ID): "legacy-startup-v1",
            },
            "qualification version dispatch differs",
        )
        expect_reject(
            lambda: qualification_profile_contract("unknown-profile"),
            "unknown qualification validator contract",
        )
        cache_fixture = {
            field: None for field in PROFILE_TIMING_MANIFEST_FIELDS
        }
        cache_fixture.update(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": ARTIFACT_TYPE,
                "status": "pass",
                "profile_id": PROFILE_ID,
                "artifact_dir": str(temp_root.resolve()),
                "source": build_source,
                "does_not_prove": DOES_NOT_PROVE,
                "pass_line": f"{PASS_PREFIX}: {temp_root.resolve()}",
            }
        )
        cache_fixture_path = temp_root / "qualification-cache-fixture.json"
        write_json(cache_fixture_path, cache_fixture)
        import runtime_vnext_r1_product_correctness as r1_correctness

        cache_context = r1_correctness._coerce_verification_context(None)
        cache_key = r1_correctness._artifact_key(
            "qualification", cache_fixture_path
        )
        cache_context.active.add(cache_key)
        expect_reject(
            lambda: verify_manifest(
                cache_fixture_path,
                verify_checkout=False,
                expected_source=build_source,
                _verification_context=cache_context,
            ),
            "cross-type R1/qualification dependency cycle",
        )
        cache_context.active.clear()
        cache_context.qualification_memo[cache_key] = {"nested": {"value": 1}}
        cached_summary = verify_manifest(
            cache_fixture_path,
            verify_checkout=False,
            expected_source=build_source,
            _verification_context=cache_context,
        )
        cached_summary["nested"]["value"] = 2
        require(
            verify_manifest(
                cache_fixture_path,
                verify_checkout=False,
                expected_source=build_source,
                _verification_context=cache_context,
            )["nested"]["value"]
            == 1,
            "qualification memo returned mutable shared state",
        )
        wrong_expected_source = copy.deepcopy(build_source)
        wrong_expected_source["git_sha"] = "0" * 40
        expect_reject(
            lambda: verify_manifest(
                cache_fixture_path,
                verify_checkout=False,
                expected_source=wrong_expected_source,
                _verification_context=cache_context,
            ),
            "cached qualification bypassed expected source",
        )

        def make_witness_case(
            name: str,
            entrypoint: str,
            *,
            stderr_text: str = "",
            raw_mutator: Any = None,
            witness_mutator: Any = None,
        ) -> Path:
            case_root = temp_root / name
            case_root.mkdir()
            stdout_path = case_root / "stdout.log"
            stderr_path = case_root / "stderr.log"
            stdout_path.write_text("startup complete\n", encoding="utf-8")
            stderr_path.write_text(stderr_text, encoding="utf-8")
            is_run = entrypoint == "run"
            profile_detail = "full" if is_run else "verify"
            profile_path = case_root / "profile.jsonl"
            profile_events = [
                {
                    "schema_version": 1,
                    "request_id": "request.startup.self-test",
                    "entrypoint": entrypoint,
                    "phase": "vnext.device_execution_span",
                    "attributes": {
                        "backend_device": "CUDA(0)",
                        "device_timing_span_kind": "reusable_executable",
                        "execution_path": "replayed",
                        "profile_detail": profile_detail,
                        "reusable_executable_fingerprint": "2" * 64,
                    },
                }
            ]
            if not is_run:
                profile_events = [
                    {
                        "schema_version": 1,
                        "request_id": "request.startup.self-test",
                        "entrypoint": "serve",
                        "phase": "vnext.device_native_work",
                        "status": "diagnostic_only",
                        "attributes": {
                            "backend_device": "CUDA(0)",
                            "device_timing_span_kind": "eager_command",
                            "execution_path": "eager",
                            "participant_request_ids": [
                                "request.startup.self-test"
                            ],
                            "profile_detail": "verify",
                        },
                    },
                    {
                        "schema_version": 1,
                        "request_id": "request.startup.self-test",
                        "entrypoint": "serve",
                        "phase": "vnext.sequence_completed",
                        "status": "ok",
                        "attributes": {
                            "backend_device": "CUDA(0)",
                            "completed_sequence_fingerprint": "3" * 64,
                            "execution_event_kind": "sequence_completed",
                            "execution_request_id": "request.startup.self-test",
                            "execution_request_origin": "startup",
                            "profile_detail": "verify",
                        },
                    },
                    {
                        "schema_version": 1,
                        "request_id": "request.startup.self-test",
                        "entrypoint": "serve",
                        "phase": "vnext.request_completed",
                        "status": "ok",
                        "attributes": {
                            "backend_device": "CUDA(0)",
                            "completed_sequence_fingerprint": "3" * 64,
                            "execution_event_kind": "request_completed",
                            "execution_request_id": "request.startup.self-test",
                            "execution_request_origin": "startup",
                            "profile_detail": "verify",
                        },
                    },
                ]
            profile_path.write_text(
                "".join(
                    json.dumps(event, sort_keys=True) + "\n"
                    for event in profile_events
                ),
                encoding="utf-8",
            )
            raw: dict[str, Any] = {
                "schema_version": 1,
                "artifact_type": RAW_EVIDENCE_TYPE,
                "source_git_sha": build_source["git_sha"],
                "source_tree_sha": build_source["git_tree_sha"],
                "backend": "cuda",
                "binary_copy": file_ref(binary_path),
                "executed_binary_path": str(binary_path.resolve()),
                "build_receipt": file_ref(build_receipt_path),
                "profile_jsonl": file_ref(profile_path),
                "command": {
                    "argv": [
                        str(binary_path.resolve()),
                        entrypoint,
                        "--profile-detail",
                        profile_detail,
                    ],
                    "returncode": 0 if is_run else None,
                    "stdout": file_ref(stdout_path),
                    "stderr": file_ref(stderr_path),
                },
                "http": None,
                "harness": {"returncode": 0, "diagnostic": None},
            }
            if is_run:
                product = {"success": True, "command_rc": 0}
                harness_summary = {"status": "pass", "failure_class": None}
            else:
                request_path = case_root / "request.json"
                write_json(request_path, {"max_tokens": 16})
                response_path = case_root / "response.json"
                write_json(
                    response_path,
                    {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {
                                    "content": "",
                                    "reasoning": "self-test thinking tokens",
                                },
                            }
                        ],
                        "usage": {"completion_tokens": 16},
                    },
                )
                diagnostic_path = case_root / "harness.log"
                diagnostic_path.write_text(
                    "response content was empty\n", encoding="utf-8"
                )
                raw["http"] = {
                    "status_code": 200,
                    "request_body": file_ref(request_path),
                    "response_body": file_ref(response_path),
                }
                raw["harness"] = {
                    "returncode": 1,
                    "diagnostic": file_ref(diagnostic_path),
                }
                product = {
                    "success": True,
                    "http_status": 200,
                    "completion_tokens": 16,
                }
                harness_summary = {
                    "status": "failed",
                    "failure_class": "non_target_configuration",
                }
            if raw_mutator is not None:
                raw_mutator(raw, case_root)
            raw_path = case_root / "raw-execution.json"
            write_json(raw_path, raw)
            check_id = (
                "cuda_run_profile_full"
                if is_run
                else "cuda_serve_profile_verify"
            )
            witness = {
                "schema_version": 1,
                "artifact_type": WITNESS_TYPE,
                "status": "target_signal_pass",
                "check_id": check_id,
                "source_git_sha": build_source["git_sha"],
                "source_tree_sha": build_source["git_tree_sha"],
                "backend": "cuda",
                "entrypoint": entrypoint,
                "profile_detail": profile_detail,
                "binary_sha256": raw_binary_sha,
                "product": product,
                "target_signal": {
                    "startup_catalog_error_count": 0,
                    "startup_reusable_execution_span_count": 1 if is_run else 0,
                    "startup_reusable_program_fingerprint_count": (
                        1 if is_run else 0
                    ),
                    "startup_verification_sequence_completed_count": (
                        0 if is_run else 1
                    ),
                    "startup_verification_request_completed_count": (
                        0 if is_run else 1
                    ),
                    "startup_verification_eager_participant_count": (
                        0 if is_run else 1
                    ),
                },
                "harness": harness_summary,
                "evidence": {"raw_execution": file_ref(raw_path)},
            }
            if witness_mutator is not None:
                witness_mutator(witness)
            witness_path = case_root / "witness.json"
            write_json(witness_path, witness)
            return witness_path

        def validate_case(path: Path, entrypoint: str) -> dict[str, Any]:
            is_run = entrypoint == "run"
            return validate_witness(
                path,
                check_id=(
                    "cuda_run_profile_full"
                    if is_run
                    else "cuda_serve_profile_verify"
                ),
                entrypoint=entrypoint,
                profile_detail="full" if is_run else "verify",
                build_source=build_source,
                binary_sha256=raw_binary_sha,
            )

        run_witness = make_witness_case("run-positive", "run")
        run_proof = validate_case(run_witness, "run")
        require(
            run_proof["raw_execution"]["command"]["returncode"] == 0
            and run_proof["raw_execution"]["target_signal"]
            == {
                "startup_catalog_error_count": 0,
                "startup_reusable_execution_span_count": 1,
                "startup_reusable_program_fingerprint_count": 1,
                "startup_verification_sequence_completed_count": 0,
                "startup_verification_request_completed_count": 0,
                "startup_verification_eager_participant_count": 0,
            },
            "self-test did not derive run raw evidence",
        )

        def forge_run_summary(witness: dict[str, Any]) -> None:
            witness["product"]["command_rc"] = 7

        expect_reject(
            lambda: validate_case(
                make_witness_case(
                    "run-forged-summary",
                    "run",
                    witness_mutator=forge_run_summary,
                ),
                "run",
            ),
            "forged run summary",
        )

        def forge_profile(raw: dict[str, Any], _: Path) -> None:
            raw["command"]["argv"][-1] = "verify"

        expect_reject(
            lambda: validate_case(
                make_witness_case(
                    "run-wrong-profile", "run", raw_mutator=forge_profile
                ),
                "run",
            ),
            "wrong raw profile detail",
        )
        expect_reject(
            lambda: validate_case(
                make_witness_case(
                    "run-startup-error", "run", stderr_text=OLD_STARTUP_ERROR
                ),
                "run",
            ),
            "startup error hidden by witness summary",
        )

        def forge_source(raw: dict[str, Any], _: Path) -> None:
            raw["source_git_sha"] = "c" * 40

        expect_reject(
            lambda: validate_case(
                make_witness_case(
                    "run-wrong-source", "run", raw_mutator=forge_source
                ),
                "run",
            ),
            "wrong raw source",
        )

        def forge_binary(raw: dict[str, Any], case_root: Path) -> None:
            other_binary = case_root / "other-ferrum"
            other_binary.write_bytes(b"different binary")
            raw["binary_copy"] = file_ref(other_binary)

        expect_reject(
            lambda: validate_case(
                make_witness_case(
                    "run-wrong-binary", "run", raw_mutator=forge_binary
                ),
                "run",
            ),
            "wrong raw binary",
        )

        serve_witness = make_witness_case("serve-positive", "serve")
        serve_proof = validate_case(serve_witness, "serve")
        require(
            serve_proof["raw_execution"]["product"]
            == {
                "success": True,
                "http_status": 200,
                "completion_tokens": 16,
            }
            and serve_proof["raw_execution"]["harness"]
            == {
                "status": "failed",
                "failure_class": "non_target_configuration",
            },
            "self-test did not derive serve raw evidence",
        )

        def drop_verify_sequence(raw: dict[str, Any], _: Path) -> None:
            profile_path = validate_ref(raw["profile_jsonl"], "self-test profile")
            events = [
                json.loads(line)
                for line in profile_path.read_text(encoding="utf-8").splitlines()
            ]
            profile_path.write_text(
                "".join(
                    json.dumps(event, sort_keys=True) + "\n"
                    for event in events
                    if event.get("phase") != "vnext.sequence_completed"
                ),
                encoding="utf-8",
            )
            raw["profile_jsonl"] = file_ref(profile_path)

        expect_reject(
            lambda: validate_case(
                make_witness_case(
                    "serve-missing-sequence",
                    "serve",
                    raw_mutator=drop_verify_sequence,
                ),
                "serve",
            ),
            "verify evidence missing sequence completion",
        )

        def make_verify_work_replayed(raw: dict[str, Any], _: Path) -> None:
            profile_path = validate_ref(raw["profile_jsonl"], "self-test profile")
            events = [
                json.loads(line)
                for line in profile_path.read_text(encoding="utf-8").splitlines()
            ]
            for event in events:
                if event.get("phase") == "vnext.device_native_work":
                    event["attributes"]["execution_path"] = "replayed"
            profile_path.write_text(
                "".join(
                    json.dumps(event, sort_keys=True) + "\n" for event in events
                ),
                encoding="utf-8",
            )
            raw["profile_jsonl"] = file_ref(profile_path)

        expect_reject(
            lambda: validate_case(
                make_witness_case(
                    "serve-replayed-native-work",
                    "serve",
                    raw_mutator=make_verify_work_replayed,
                ),
                "serve",
            ),
            "verify evidence used replayed native work",
        )

        def forge_completion_summary(witness: dict[str, Any]) -> None:
            witness["product"]["completion_tokens"] = 99

        expect_reject(
            lambda: validate_case(
                make_witness_case(
                    "serve-forged-completion",
                    "serve",
                    witness_mutator=forge_completion_summary,
                ),
                "serve",
            ),
            "forged serve completion count",
        )

        def forge_http_status(raw: dict[str, Any], _: Path) -> None:
            raw["http"]["status_code"] = 503

        expect_reject(
            lambda: validate_case(
                make_witness_case(
                    "serve-wrong-http", "serve", raw_mutator=forge_http_status
                ),
                "serve",
            ),
            "forged serve HTTP success",
        )

        def remove_reusable_reachability(raw: dict[str, Any], case_root: Path) -> None:
            empty_profile = case_root / "empty-profile.jsonl"
            empty_profile.write_text("", encoding="utf-8")
            raw["profile_jsonl"] = file_ref(empty_profile)

        expect_reject(
            lambda: validate_case(
                make_witness_case(
                    "run-no-reusable-reachability",
                    "run",
                    raw_mutator=remove_reusable_reachability,
                ),
                "run",
            ),
            "missing startup reusable reachability",
        )

        def fatal_serve_diagnostic(raw: dict[str, Any], case_root: Path) -> None:
            diagnostic = case_root / "fatal-harness.log"
            diagnostic.write_text("CUDA OOM while serving\n", encoding="utf-8")
            raw["harness"]["diagnostic"] = file_ref(diagnostic)

        expect_reject(
            lambda: validate_case(
                make_witness_case(
                    "serve-fatal-diagnostic",
                    "serve",
                    raw_mutator=fatal_serve_diagnostic,
                ),
                "serve",
            ),
            "fatal serve harness diagnostic",
        )

        # Hostile execution-closure fixtures: formal profile timing must be
        # derived from one bounded ferrum process and its exact copied outputs.
        execution_root = temp_root / "execution-closure-v2"
        execution_root.mkdir()
        model_contract = m1_cuda_model_contract(build_source)
        product_request_id = "request.product.fixture"
        profile_path = execution_root / "profile.jsonl"
        profile_events = copy.deepcopy(cuda_profile_fixture)
        for event in profile_events:
            event["timestamp"] = "2026-08-13T00:00:05+00:00"
            event["model"] = model_contract["repo"]
            event["entrypoint"] = "run"
            attributes = event.setdefault("attributes", {})
            attributes["profile_detail"] = "full"
            attributes["backend_device"] = "CUDA(0)"
        profile_events.append(
            {
                "schema_version": 1,
                "event_id": "startup-reusable-selftest",
                "timestamp": "2026-08-13T00:00:05+00:00",
                "request_id": "request.startup.self-test",
                "correlation_id": "request.startup.self-test",
                "entrypoint": "run",
                "model": model_contract["repo"],
                "phase": "vnext.device_execution_span",
                "status": "diagnostic_only",
                "attributes": {
                    "backend_device": "CUDA(0)",
                    "profile_detail": "full",
                    "device_timing_span_kind": "reusable_executable",
                    "execution_path": "replayed",
                    "reusable_executable_fingerprint": "9" * 64,
                },
            }
        )
        profile_path.write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in profile_events),
            encoding="utf-8",
        )
        config_path = execution_root / "effective-config.json"
        write_json(config_path, {"backend": "cuda", "entries": []})
        dump_root = execution_root / "request-dump"
        import request_replay_bundle_gate as replay_gate

        replay_gate.make_bundle(dump_root)
        fixture_bundle = dump_root / "req-fixture"
        bundle_root = dump_root / product_request_id
        fixture_bundle.rename(bundle_root)
        for json_path in bundle_root.glob("*.json"):
            body = read_json(json_path, f"self-test {json_path.name}")
            if "request_id" in body:
                body["request_id"] = product_request_id
            if json_path.name == "request.json":
                body.update(
                    {
                        "entrypoint": "run",
                        "backend": "actual",
                        "actual_model_smoke": True,
                        "sanitized": True,
                        "model": "/models/m1",
                        "profile_detail": "full",
                        "profile_sample_rate": 1.0,
                        "l0_only": False,
                    }
                )
            elif json_path.name == "backend_selection.json":
                body.update(
                    {
                        "backend": "actual",
                        "model": "/models/m1",
                        "actual_model_smoke": True,
                    }
                )
            elif json_path.name == "sampling_params.json":
                body["sampling_params"] = {"max_tokens": 4, "temperature": 0.0}
            elif json_path.name == "runtime_effective_config.json":
                body.update(
                    {
                        "entrypoint": "run",
                        "profile_detail": "full",
                        "profile_sample_rate": 1.0,
                        "profile_jsonl": str(profile_path.resolve()),
                        "memory_profile_jsonl": None,
                        "scheduler_trace_jsonl": None,
                        "request_dump_dir": str(dump_root.resolve()),
                        "sanitized": True,
                    }
                )
            elif json_path.name == "replay.command.json":
                replay_argv = [
                    "cargo", "run", "-p", "ferrum-cli", "--", "run",
                    "synthetic/no-weight", "--profile-detail", "full",
                    "--profile-sample-rate", "1", "--profile-jsonl",
                    str(profile_path.resolve()), "--request-dump-dir",
                    str(dump_root.resolve()),
                ]
                engine_argv = [
                    "cargo", "run", "-p", "ferrum-cli", "--",
                    "replay-bundle", str(bundle_root.resolve()), "--out",
                    str((bundle_root / "engine_replay").resolve()), "--json",
                ]
                body.update(
                    {
                        "entrypoint": "run",
                        "argv": replay_argv,
                        "command": producer_command(replay_argv),
                        "bundle_dir": str(bundle_root.resolve()),
                        "sanitized": True,
                        "engine_replay": {
                            "mode": "bundle_offline",
                            "requires_http_server": False,
                            "argv": engine_argv,
                            "command": producer_command(engine_argv),
                        },
                    }
                )
            write_json(json_path, body)
        request_body = read_json(bundle_root / "request.json", "self-test request")
        request_body["replay_command"] = read_json(
            bundle_root / "replay.command.json", "self-test replay"
        )["command"]
        write_json(bundle_root / "request.json", request_body)

        def directory_closure(root: Path) -> dict[str, Any]:
            rows = [
                {
                    "path": item.relative_to(root).as_posix(),
                    "size_bytes": item.stat().st_size,
                    "sha256": sha256(item),
                }
                for item in sorted(root.rglob("*"), key=lambda value: value.as_posix())
                if item.is_file()
            ]
            return {
                "path": str(root.resolve()),
                "file_count": len(rows),
                "files": rows,
                "closure_sha256": sha256_bytes(
                    json.dumps(
                        rows,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("utf-8")
                ),
            }

        product_stdout = execution_root / "product.stdout.jsonl"
        product_stdout.write_text(
            "\n".join(
                json.dumps(row, sort_keys=True)
                for row in (
                    {
                        "event": "ready",
                        "session_id": "session-selftest",
                        "model": model_contract["repo"],
                        "requested_model": "/models/m1",
                        "resolved_model": model_contract["repo"],
                        "backend": "CUDA(0)",
                    },
                    {
                        "event": "user",
                        "session_id": "session-selftest",
                        "request_id": product_request_id,
                        "content": "public fixture",
                    },
                    {
                        "event": "assistant",
                        "session_id": "session-selftest",
                        "request_id": product_request_id,
                        "content": "PROFILE-OK",
                        "n_tokens": 2,
                    },
                    {"event": "exit", "session_id": "session-selftest"},
                )
            )
            + "\n",
            encoding="utf-8",
        )
        product_stderr = execution_root / "product.stderr.log"
        product_stderr.write_text("", encoding="utf-8")
        product_command = [
            str(binary_path.resolve()),
            "run",
            "/models/m1",
            "--backend",
            "cuda",
            "--prompt",
            "public fixture",
            "--max-tokens",
            "4",
            "--disable-thinking",
            "--temperature",
            "0",
            "--repeat-penalty",
            "1.0",
            "--output-format",
            "jsonl",
            "--semantic-source",
            "/models/m1",
            "--tokenizer-source",
            "/models/m1",
            "--profile-detail",
            "full",
            "--effective-config-json",
            str(config_path.resolve()),
            "--profile-sample-rate",
            "1.0",
            "--profile-jsonl",
            str(profile_path.resolve()),
            "--request-dump-dir",
            str(dump_root.resolve()),
        ]

        def bounded_receipt(
            receipt_path: Path,
            command: list[str],
            stdout_path: Path,
            stderr_path: Path,
            started_at: str,
            ended_at: str,
        ) -> None:
            write_json(
                receipt_path,
                {
                    "schema": "ferrum.bounded-command-receipt.v1",
                    "status": "pass",
                    "reason": "command_completed",
                    "rc": 0,
                    "command": ["/usr/bin/env", "-i", "PATH=/usr/bin", *command],
                    "limits": {
                        "wall_timeout_seconds": 600,
                        "max_processes": 8,
                        "max_group_threads": 64,
                        "max_per_process_threads": 64,
                    },
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "violation": None,
                    "cleanup": {"process_group_gone": True},
                    "termination": {"errors": [], "signals": []},
                    "sampling_error_count": 0,
                    "sampling_errors": [],
                    "stdout": file_ref(stdout_path),
                    "stderr": file_ref(stderr_path),
                },
            )

        product_receipt = execution_root / "product.bounded.receipt.json"
        bounded_receipt(
            product_receipt,
            product_command,
            product_stdout,
            product_stderr,
            "2026-08-13T00:00:00+00:00",
            "2026-08-13T00:00:10+00:00",
        )
        hardware_stdout = execution_root / "hardware.stdout.log"
        hardware_stdout.write_text(
            "0, NVIDIA GeForce RTX 4090, "
            "GPU-12345678-1234-5678-9abc-def012345678, 24564, 570.00\n",
            encoding="utf-8",
        )
        hardware_stderr = execution_root / "hardware.stderr.log"
        hardware_stderr.write_text("", encoding="utf-8")
        hardware_receipt = execution_root / "hardware.bounded.receipt.json"
        hardware_command = [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
        bounded_receipt(
            hardware_receipt,
            hardware_command,
            hardware_stdout,
            hardware_stderr,
            "2026-08-12T23:59:00+00:00",
            "2026-08-12T23:59:01+00:00",
        )
        vast_instance_path = execution_root / "vast-instance.json"
        write_json(
            vast_instance_path,
            {
                "instances": {
                    "id": 42,
                    "cur_state": "running",
                    "actual_status": "running",
                    "num_gpus": 1,
                    "gpu_name": "RTX 4090",
                    "gpu_ram": 24564,
                    "driver_version": "570.00",
                }
            },
        )
        raw_v2 = {
            "schema_version": 2,
            "artifact_type": RAW_EVIDENCE_TYPE,
            "source_git_sha": build_source["git_sha"],
            "source_tree_sha": build_source["git_tree_sha"],
            "backend": "cuda",
            "vast_instance_metadata": file_ref(vast_instance_path),
            "binary_copy": file_ref(binary_path),
            "executed_binary": {
                "path": str(binary_path.resolve()),
                "sha256": raw_binary_sha,
                "size_bytes": binary_path.stat().st_size,
            },
            "build_receipt": file_ref(build_receipt_path),
            "product_receipt": file_ref(product_receipt),
            "product_stdout": file_ref(product_stdout),
            "product_stderr": file_ref(product_stderr),
            "hardware_receipt": file_ref(hardware_receipt),
            "hardware_stdout": file_ref(hardware_stdout),
            "hardware_stderr": file_ref(hardware_stderr),
            "execution_outputs": {
                "profile_jsonl": {
                    "executed_path": str(profile_path.resolve()),
                    "copy": file_ref(profile_path),
                },
                "effective_config": {
                    "executed_path": str(config_path.resolve()),
                    "copy": file_ref(config_path),
                },
                "request_dump": {
                    "executed_path": str(dump_root.resolve()),
                    "copy": directory_closure(dump_root),
                },
            },
            "model_identity": {
                "model_argument": "/models/m1",
                "semantic_argument": "/models/m1",
                "repo": model_contract["repo"],
                "revision": model_contract["revision"],
                "model_files": model_contract["model_files"],
                "semantic_files": model_contract["semantic_files"],
                "lock": model_contract["lock"],
            },
        }
        model_lock_validation = execution_root / "model-lock-validation.json"
        write_json(
            model_lock_validation,
            {
                "status": "pass",
                "snapshot_path": "/models/m1",
                "repo": model_contract["repo"],
                "revision": model_contract["revision"],
                "files": model_contract["unique_files"],
            },
        )
        raw_v2["model_lock_validation"] = file_ref(model_lock_validation)
        raw_v2_path = execution_root / "raw-v2.json"
        write_json(raw_v2_path, raw_v2)
        derived_v2 = validate_raw_execution(
            raw_v2_path,
            check_id="cuda_run_profile_full",
            entrypoint="run",
            profile_detail="full",
            build_source=build_source,
            binary_sha256=raw_binary_sha,
            require_timing_metrics=True,
            expected_build_receipt=file_ref(build_receipt_path),
        )
        require(
            derived_v2["witness_hardware"]["uuid"]
            == "GPU-12345678-1234-5678-9abc-def012345678"
            and derived_v2["witness_hardware"]["instance_id"] == 42
            and derived_v2["model"]["repo"] == model_contract["repo"]
            and derived_v2["target_signal"]["product_profile_event_count"] > 0,
            "formal execution closure was not mechanically derived",
        )

        def reject_raw_v2(name: str, value: dict[str, Any]) -> None:
            hostile_path = execution_root / f"{name}.json"
            write_json(hostile_path, value)
            expect_reject(
                lambda: validate_raw_execution(
                    hostile_path,
                    check_id="cuda_run_profile_full",
                    entrypoint="run",
                    profile_detail="full",
                    build_source=build_source,
                    binary_sha256=raw_binary_sha,
                    require_timing_metrics=True,
                    expected_build_receipt=file_ref(build_receipt_path),
                ),
                name,
            )

        hostile_binary = copy.deepcopy(raw_v2)
        other_binary = execution_root / "unrelated-ferrum"
        other_binary.write_bytes(b"unrelated")
        hostile_binary["binary_copy"] = file_ref(other_binary)
        reject_raw_v2("formal-swapped-binary", hostile_binary)

        hostile_receipt = copy.deepcopy(raw_v2)
        bad_product_receipt = read_json(product_receipt, "self-test product receipt")
        bad_product_receipt["cleanup"] = {"process_group_gone": False}
        bad_product_receipt_path = execution_root / "bad-product-receipt.json"
        write_json(bad_product_receipt_path, bad_product_receipt)
        hostile_receipt["product_receipt"] = file_ref(bad_product_receipt_path)
        reject_raw_v2("formal-unbounded-product", hostile_receipt)

        hostile_hardware = copy.deepcopy(raw_v2)
        changed_hardware = execution_root / "changed-hardware.stdout"
        changed_hardware.write_text(
            "0, NVIDIA GeForce RTX 4090, GPU-other, 24564, 570.00\n",
            encoding="utf-8",
        )
        hostile_hardware["hardware_stdout"] = file_ref(changed_hardware)
        reject_raw_v2("formal-hardware-log-substitution", hostile_hardware)

        def reject_hardware_shape(
            name: str,
            stdout_text: str,
            *,
            gpu_name: str,
            num_gpus: int,
            gpu_ram: int = 24564,
        ) -> None:
            variant_stdout = execution_root / f"{name}.stdout"
            variant_stderr = execution_root / f"{name}.stderr"
            variant_receipt = execution_root / f"{name}.receipt.json"
            variant_instance = execution_root / f"{name}.vast.json"
            variant_stdout.write_text(stdout_text, encoding="utf-8")
            variant_stderr.write_text("", encoding="utf-8")
            bounded_receipt(
                variant_receipt,
                hardware_command,
                variant_stdout,
                variant_stderr,
                "2026-08-12T23:59:00+00:00",
                "2026-08-12T23:59:01+00:00",
            )
            write_json(
                variant_instance,
                {
                    "instances": {
                        "id": 43,
                        "cur_state": "running",
                        "actual_status": "running",
                        "num_gpus": num_gpus,
                        "gpu_name": gpu_name,
                        "gpu_ram": gpu_ram,
                        "driver_version": "570.00",
                    }
                },
            )
            hostile = copy.deepcopy(raw_v2)
            hostile["hardware_receipt"] = file_ref(variant_receipt)
            hostile["hardware_stdout"] = file_ref(variant_stdout)
            hostile["hardware_stderr"] = file_ref(variant_stderr)
            hostile["vast_instance_metadata"] = file_ref(variant_instance)
            reject_raw_v2(name, hostile)

        reject_hardware_shape(
            "formal-malformed-gpu-uuid",
            "0, NVIDIA GeForce RTX 4090, GPU-forged, 24564, 570.00\n",
            gpu_name="RTX 4090",
            num_gpus=1,
        )
        reject_hardware_shape(
            "formal-multiple-gpus",
            "0, NVIDIA GeForce RTX 4090, "
            "GPU-12345678-1234-5678-9abc-def012345678, 24564, 570.00\n"
            "1, NVIDIA GeForce RTX 4090, "
            "GPU-87654321-4321-8765-cba9-876543210fed, 24564, 570.00\n",
            gpu_name="RTX 4090",
            num_gpus=2,
            gpu_ram=49128,
        )
        reject_hardware_shape(
            "formal-non-4090",
            "0, NVIDIA A100-SXM4-80GB, "
            "GPU-12345678-1234-5678-9abc-def012345678, 81920, 570.00\n",
            gpu_name="A100-SXM4-80GB",
            num_gpus=1,
            gpu_ram=81920,
        )

        hostile_model = copy.deepcopy(raw_v2)
        hostile_model["model_identity"]["revision"] = "f" * 40
        reject_raw_v2("formal-model-lock-substitution", hostile_model)

        hostile_config = copy.deepcopy(raw_v2)
        bad_config = execution_root / "bad-effective-config.json"
        write_json(bad_config, {"backend": "metal"})
        hostile_config["execution_outputs"]["effective_config"]["copy"] = file_ref(
            bad_config
        )
        reject_raw_v2("formal-effective-backend-substitution", hostile_config)

        hostile_profile = copy.deepcopy(raw_v2)
        unrelated_profile = execution_root / "unrelated-profile.jsonl"
        unrelated_events = copy.deepcopy(profile_events)
        measured = next(
            event
            for event in unrelated_events
            if event.get("attributes", {}).get("device_timing_status") == "measured"
        )
        measured["request_id"] = "request.foreign"
        measured["correlation_id"] = "request.foreign"
        measured.get("attributes", {}).pop("execution_request_id", None)
        unrelated_profile.write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in unrelated_events),
            encoding="utf-8",
        )
        hostile_profile["execution_outputs"]["profile_jsonl"]["copy"] = file_ref(
            unrelated_profile
        )
        reject_raw_v2("formal-unrelated-measured-row", hostile_profile)

        expect_reject(
            lambda: validate_raw_execution(
                validate_ref(
                    read_json(run_witness, "legacy witness")["evidence"]["raw_execution"],
                    "legacy raw",
                ),
                check_id="cuda_run_profile_full",
                entrypoint="run",
                profile_detail="full",
                build_source=build_source,
                binary_sha256=raw_binary_sha,
                require_timing_metrics=True,
                expected_build_receipt=file_ref(build_receipt_path),
            ),
            "formal timing accepted legacy raw schema",
        )
    with tempfile.TemporaryDirectory(
        prefix="ferrum-qualification-closure-selftest-"
    ) as temporary:
        root = Path(temporary)
        source = source_at(git_text("rev-parse", "HEAD"))
        unit_path = root / "unit.receipt.json"
        unit_stdout = root / "unit.stdout.log"
        unit_stderr = root / "unit.stderr.log"
        unit_stdout.write_text(LEGACY_UNIT_TEST_PASS_LINE + "\n", encoding="utf-8")
        unit_stderr.write_text("", encoding="utf-8")
        unit_receipt = {
            "schema": "ferrum.bounded-command-receipt.v1",
            "status": "pass",
            "rc": 0,
            "reason": "command_completed",
            "command": legacy_exact_unit_test_command(source),
            "cwd": str(REPO_ROOT),
            "violation": None,
            "cleanup": {"process_group_gone": True},
            "termination": {"errors": [], "signals": []},
            "sampling_error_count": 0,
            "sampling_errors": [],
            "stdout": file_ref(unit_stdout),
            "stderr": file_ref(unit_stderr),
        }
        write_json(unit_path, unit_receipt)
        require(
            validate_unit_receipt(unit_path, source)["source"] == source,
            "self-test unit source binding differs",
        )
        forged_unit = copy.deepcopy(unit_receipt)
        forged_unit["command"][-1] = "f" * 40
        forged_unit_path = root / "unit-forged.receipt.json"
        write_json(forged_unit_path, forged_unit)
        expect_reject(
            lambda: validate_unit_receipt(forged_unit_path, source),
            "unit receipt stale source binding",
        )

        def focused_receipt(
            name: str, command: list[str], pass_line: str
        ) -> Path:
            stdout = root / f"{name}.stdout.log"
            stderr = root / f"{name}.stderr.log"
            receipt_path = root / f"{name}.receipt.json"
            stdout.write_text(pass_line + "\n", encoding="utf-8")
            stderr.write_text("", encoding="utf-8")
            write_json(
                receipt_path,
                {
                    "schema": "ferrum.bounded-command-receipt.v1",
                    "status": "pass",
                    "rc": 0,
                    "reason": "command_completed",
                    "command": command,
                    "cwd": str(REPO_ROOT),
                    "violation": None,
                    "cleanup": {"process_group_gone": True},
                    "termination": {"errors": [], "signals": []},
                    "sampling_error_count": 0,
                    "sampling_errors": [],
                    "stdout": file_ref(stdout),
                    "stderr": file_ref(stderr),
                },
            )
            return receipt_path

        exact_path = focused_receipt(
            "exact-contracts",
            profile_timing_exact_contracts_command(source),
            EXACT_CONTRACTS_PASS_LINE,
        )
        collector_path = focused_receipt(
            "collector-selftest",
            profile_collector_selftest_command(source),
            PROFILE_COLLECTOR_SELFTEST_PASS_LINE,
        )
        require(
            validate_profile_timing_exact_contracts_receipt(exact_path, source)[
                "check_id"
            ]
            == "profile_timing_exact_contracts"
            and validate_profile_collector_selftest_receipt(
                collector_path, source
            )["check_id"]
            == "profile_collector_selftest",
            "focused profile receipt validation differs",
        )

        control_stdout = root / "control.stdout.log"
        control_stderr = root / "control.stderr.log"
        control_stdout.write_text(
            "\n".join(
                [
                    PLANNER_SELFTEST_PASS_LINE,
                    SELFTEST_PASS_LINE,
                    R1_SELFTEST_PASS_LINE,
                    HOST_SUSPEND_SELFTEST_PASS_LINE,
                    CONTROL_SELFTEST_PASS_LINE,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        control_stderr.write_text("", encoding="utf-8")
        control_path = root / "control.receipt.json"
        control_receipt = {
            "schema": "ferrum.bounded-command-receipt.v1",
            "status": "pass",
            "rc": 0,
            "reason": "command_completed",
            "command": control_selftest_command(source),
            "cwd": str(REPO_ROOT),
            "violation": None,
            "cleanup": {"process_group_gone": True},
            "termination": {"errors": [], "signals": []},
            "sampling_error_count": 0,
            "sampling_errors": [],
            "stdout": file_ref(control_stdout),
            "stderr": file_ref(control_stderr),
        }
        write_json(control_path, control_receipt)
        require(
            validate_control_gate_receipt(control_path, source)["gates"]
            == ["planner_selftest", "release_validator_selftest"],
            "self-test control gate proof differs",
        )
        forged_control = copy.deepcopy(control_receipt)
        forged_control["command"][-1] = "e" * 40
        forged_control_path = root / "control-forged.receipt.json"
        write_json(forged_control_path, forged_control)
        expect_reject(
            lambda: validate_control_gate_receipt(forged_control_path, source),
            "control receipt stale source binding",
        )
    print(SELFTEST_PASS_LINE)
    return 0


def control_self_test(expected_source_sha: str) -> int:
    require(
        GIT_SHA_RE.fullmatch(expected_source_sha) is not None,
        "control self-test expected source SHA is invalid",
    )
    source = current_source()
    require(
        source["git_sha"] == expected_source_sha,
        "control self-test source differs from the requested qualification source",
    )
    validate_canonical_input_refs(canonical_input_refs(source), source)

    require(self_test() == 0, "qualification focused self-test failed")

    rule_config = plan_gates.load_rule_config(RULES_PATH)
    planner_result = plan_gates.run_selftest(
        rule_config["rules"],
        rule_config["qualification_profiles"],
        FIXTURES_PATH,
    )
    require(
        planner_result.get("status") == "pass",
        f"change-impact planner focused self-test failed: {planner_result.get('failures')}",
    )
    print(PLANNER_SELFTEST_PASS_LINE)

    import run_gate
    import runtime_vnext_baseline_scenarios as baseline_scenarios
    import runtime_vnext_r1_product_correctness as r1_correctness

    require(r1_correctness.self_test() == 0, "R1 cumulative focused self-test failed")
    require(
        baseline_scenarios.self_test_host_suspend_assembler() == 0,
        "historical host-suspend focused self-test failed",
    )

    prior = Path("/tmp/ferrum-control-selftest-prior-r1.json")
    qualification = Path("/tmp/ferrum-control-selftest-impact-qualification.json")
    out = Path("/tmp/ferrum-control-selftest-r1-out")
    route_args = SimpleNamespace(
        lane="vnext-r1",
        prior_r1=prior,
        impact_qualification=qualification,
        r0=None,
        m1_cuda=None,
        m1_metal=None,
        m2_cuda=None,
        m2_metal=None,
        m3_cuda=None,
        m3_metal=None,
        llama_cuda=None,
        llama_metal=None,
    )
    route = run_gate.build_lane_command(route_args, out)
    expected_route = [
        os.sys.executable,
        "scripts/release/runtime_vnext_r1_product_correctness.py",
        "--prior-r1",
        str(prior.resolve()),
        "--impact-qualification",
        str(qualification.resolve()),
        "--out",
        str(out),
    ]
    require(
        route.cmd == expected_route
        and route.provenance_kind == "vnext-r1"
        and route.expected_child_pass_line
        == f"FERRUM RUNTIME VNEXT R1 PRODUCT CORRECTNESS PASS: {out}",
        "unified gate cumulative R1 route differs",
    )
    require(current_source() == source, "control self-test source changed during execution")
    print(CONTROL_SELFTEST_PASS_LINE)
    return 0


def run_exact_unit_test(expected_source_sha: str) -> int:
    require(
        GIT_SHA_RE.fullmatch(expected_source_sha) is not None,
        "exact unit expected source SHA is invalid",
    )
    source = current_source()
    require(
        source["git_sha"] == expected_source_sha,
        "exact unit source differs from the requested qualification source",
    )
    command = [
        "cargo",
        "test",
        "-p",
        "ferrum-models",
        "--lib",
        LEGACY_EXACT_UNIT_FILTER,
        "--",
        "--exact",
        "--test-threads=1",
    ]
    process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.stdout:
        print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, end="", file=os.sys.stderr)
    require(process.returncode == 0, "exact unit cargo test failed")
    combined = process.stdout + "\n" + process.stderr
    require(
        "running 1 test" in combined
        and f"test {LEGACY_EXACT_UNIT_FILTER} ... ok" in combined
        and "test result: ok. 1 passed; 0 failed" in combined,
        "exact unit cargo output did not prove 1/1 PASS",
    )
    require(current_source() == source, "exact unit source changed during execution")
    print(LEGACY_UNIT_TEST_PASS_LINE)
    return 0


def run_profile_timing_exact_contracts(expected_source_sha: str) -> int:
    source = current_source()
    require(
        source["git_sha"] == expected_source_sha,
        "profile timing exact-contract source differs",
    )
    commands = [
        [
            "cargo", "test", "-p", "ferrum-engine", "--lib",
            "continuous_engine::tests::plan_runtime_profile_records_only_explicit_decode_host_stages",
            "--", "--exact", "--test-threads=1",
        ],
        [
            "cargo", "test", "-p", "ferrum-models", "--lib",
            "executor::vnext_executor::tests::monotonic_wall_anchor_uses_sample_midpoint_and_bounds_full_capture_span",
            "--", "--exact", "--test-threads=1",
        ],
        [
            "cargo", "test", "-p", "ferrum-models", "--lib",
            "executor::vnext_executor::tests::journal_clock_anchor_is_kernel_profile_only",
            "--", "--exact", "--test-threads=1",
        ],
        [
            "cargo", "test", "-p", "ferrum-interfaces", "--test",
            "vnext_event_execution_contract_tests",
            "request_accepted_clock_anchor_is_typed_and_legacy_none_remains_valid",
            "--", "--exact", "--test-threads=1",
        ],
        [
            "cargo", "test", "-p", "ferrum-types", "--lib",
            "observability_profile::tests::engine_token_timing_preserves_exact_commit_intervals",
            "--", "--exact", "--test-threads=1",
        ],
    ]
    for command in commands:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.stdout:
            print(process.stdout, end="")
        if process.stderr:
            print(process.stderr, end="", file=os.sys.stderr)
        combined = process.stdout + "\n" + process.stderr
        require(
            process.returncode == 0
            and "running 1 test" in combined
            and "test result: ok. 1 passed; 0 failed" in combined,
            f"profile timing exact contract failed: {' '.join(command)}",
        )
    require(current_source() == source, "profile timing exact-contract source changed")
    print(EXACT_CONTRACTS_PASS_LINE)
    return 0


def run_profile_collector_selftest(expected_source_sha: str) -> int:
    source = current_source()
    require(
        source["git_sha"] == expected_source_sha,
        "profile collector self-test source differs",
    )
    collector = REPO_ROOT / "scripts/release/runtime_vnext_r2_profile_collector.py"
    for command in (
        [os.sys.executable, "-m", "py_compile", str(collector)],
        [os.sys.executable, str(collector), "--self-test"],
    ):
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.stdout:
            print(process.stdout, end="")
        if process.stderr:
            print(process.stderr, end="", file=os.sys.stderr)
        require(process.returncode == 0, f"profile collector check failed: {command}")
        if command[-1] == "--self-test":
            require(
                process.stdout.splitlines()[-1:]
                == ["FERRUM RUNTIME VNEXT R2 PROFILE COLLECTOR SELFTEST PASS"],
                "profile collector exact self-test PASS line is missing",
            )
    require(current_source() == source, "profile collector self-test source changed")
    print(PROFILE_COLLECTOR_SELFTEST_PASS_LINE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-r1", type=Path)
    parser.add_argument("--profile-id", default=PROFILE_ID)
    parser.add_argument("--candidate-cuda-binary", type=Path)
    parser.add_argument("--candidate-cuda-build-receipt", type=Path)
    parser.add_argument("--exact-contracts-receipt", type=Path)
    parser.add_argument("--profile-collector-selftest-receipt", type=Path)
    parser.add_argument("--control-gate-receipt", type=Path)
    parser.add_argument("--run-full-witness", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--control-self-test", action="store_true")
    parser.add_argument("--run-exact-unit-test", action="store_true")
    parser.add_argument("--run-profile-timing-exact-contracts", action="store_true")
    parser.add_argument("--run-profile-collector-selftest", action="store_true")
    parser.add_argument("--expected-source-sha")
    args = parser.parse_args()
    try:
        if args.self_test:
            return self_test()
        if args.control_self_test:
            require(
                args.expected_source_sha is not None,
                "--control-self-test requires --expected-source-sha",
            )
            return control_self_test(args.expected_source_sha)
        if args.run_exact_unit_test:
            require(
                args.expected_source_sha is not None,
                "--run-exact-unit-test requires --expected-source-sha",
            )
            return run_exact_unit_test(args.expected_source_sha)
        if args.run_profile_timing_exact_contracts:
            require(
                args.expected_source_sha is not None,
                "--run-profile-timing-exact-contracts requires --expected-source-sha",
            )
            return run_profile_timing_exact_contracts(args.expected_source_sha)
        if args.run_profile_collector_selftest:
            require(
                args.expected_source_sha is not None,
                "--run-profile-collector-selftest requires --expected-source-sha",
            )
            return run_profile_collector_selftest(args.expected_source_sha)
        if args.verify_manifest is not None:
            verified = verify_manifest(args.verify_manifest, verify_checkout=True)
            print(f"{PASS_PREFIX}: {Path(verified['manifest']['path']).parent}")
            return 0
        required = {
            "prior_r1": args.prior_r1,
            "candidate_cuda_binary": args.candidate_cuda_binary,
            "candidate_cuda_build_receipt": args.candidate_cuda_build_receipt,
            "exact_contracts_receipt": args.exact_contracts_receipt,
            "profile_collector_selftest_receipt": args.profile_collector_selftest_receipt,
            "control_gate_receipt": args.control_gate_receipt,
            "run_full_witness": args.run_full_witness,
            "out": args.out,
        }
        missing = [key for key, value in required.items() if value is None]
        require(not missing, f"missing required inputs: {missing}")
        assert all(value is not None for value in required.values())
        print(
            build(
                prior_r1=args.prior_r1,
                profile_id=args.profile_id,
                candidate_cuda_binary=args.candidate_cuda_binary,
                candidate_cuda_build_receipt=args.candidate_cuda_build_receipt,
                exact_contracts_receipt=args.exact_contracts_receipt,
                profile_collector_selftest_receipt=args.profile_collector_selftest_receipt,
                control_gate_receipt=args.control_gate_receipt,
                run_full_witness=args.run_full_witness,
                out=args.out,
            )
        )
        return 0
    except (OSError, QualificationError, RuntimeError, ValueError) as error:
        print(f"FERRUM RUNTIME VNEXT CHANGE IMPACT QUALIFICATION FAIL: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
