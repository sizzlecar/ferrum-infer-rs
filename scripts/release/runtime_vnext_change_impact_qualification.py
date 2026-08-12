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
PROFILE_ID = "vnext-reusable-startup-diagnostic-observability"
PASS_PREFIX = "FERRUM RUNTIME VNEXT CHANGE IMPACT QUALIFICATION PASS"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT CHANGE IMPACT QUALIFICATION SELFTEST PASS"
CONTROL_SELFTEST_PASS_LINE = (
    "FERRUM RUNTIME VNEXT CHANGE IMPACT CONTROL SELFTEST PASS"
)
UNIT_TEST_PASS_LINE = (
    "FERRUM RUNTIME VNEXT CHANGE IMPACT EXACT UNIT PASS"
)
EXACT_UNIT_FILTER = (
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
EXECUTOR_PATH = "crates/ferrum-models/src/executor/vnext_executor.rs"
EXPECTED_CHECKS = [
    "cuda_run_profile_full",
    "cuda_serve_profile_verify",
    "executor_unit_reusable_program_identity",
]
EXPECTED_SCOPES = [
    {"backend": "cuda", "entrypoint": "run", "profile_detail": "full"},
    {"backend": "cuda", "entrypoint": "serve", "profile_detail": "verify"},
]
EXPECTED_CONTROL_GATES = [
    "docs_review",
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


def exact_unit_test_command(source: dict[str, Any]) -> list[str]:
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
        if path != EXECUTOR_PATH:
            rejected.append(path)
            continue
        try:
            before = git_file_bytes(build_source["git_sha"], path).decode("utf-8")
            after = git_file_bytes(qualified_source["git_sha"], path).decode("utf-8")
        except UnicodeDecodeError as error:
            raise QualificationError("executor source is not UTF-8") from error
        before_product, before_error = plan_gates.production_text(before, True)
        after_product, after_error = plan_gates.production_text(after, True)
        require(
            before_product is not None and after_product is not None,
            "executor test boundary is unsafe: "
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


def prior_r1_summary(path: Path) -> dict[str, Any]:
    manifest = read_json(path, "prior R1 manifest")
    require(
        manifest.get("artifact_type") == "runtime_vnext_r1_product_correctness_manifest"
        and manifest.get("checkpoint_id") == "R1"
        and manifest.get("status") == "pass"
        and manifest.get("canonical") is True,
        "prior R1 is not a canonical PASS",
    )
    dependencies = manifest.get("dependencies")
    require(
        isinstance(dependencies, dict)
        and set(dependencies) == {"r0", "matrices", "llama_dense", "acceptance"},
        "prior R1 must be the original full aggregate",
    )
    source = normalize_source(manifest.get("source"), "prior R1")
    acceptance = manifest.get("acceptance")
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
    reused_cells = derive_reused_cells(acceptance)
    profile_flag_count = 0
    hidden_profile_env_count = 0
    matrices = dependencies.get("matrices")
    llamas = dependencies.get("llama_dense")
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
    return {
        "source": source,
        "acceptance": copy.deepcopy(acceptance),
        "backend_binary_sha256": copy.deepcopy(binaries),
        "reused_cells": reused_cells,
        "reachability": {
            "matrix_case_count": 1867,
            "llama_scenario_count": 6,
            "profile_flag_count": profile_flag_count,
            "hidden_profile_env_count": hidden_profile_env_count,
            "mode": "profile_off",
        },
    }


def validate_candidate_build(
    binary_path: Path,
    receipt_path: Path,
    qualified_source: dict[str, Any],
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
    projection = release_product_projection(build_source, qualified_source)
    return {
        "binary": binary,
        "build_receipt": file_ref(receipt_path),
        "build_source": build_source,
        "release_product_projection": projection,
        "binary_sha256": binary_sha,
    }


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
        receipt.get("command") == exact_unit_test_command(qualified_source),
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
        stdout_lines.count(UNIT_TEST_PASS_LINE) == 1
        and stdout_lines[-1:] == [UNIT_TEST_PASS_LINE],
        "exact unit source-bound PASS line is missing",
    )
    return {
        "check_id": "executor_unit_reusable_program_identity",
        "receipt": file_ref(path),
        "stdout": file_ref(stdout_path),
        "stderr": file_ref(stderr_path),
        "pass_line": UNIT_TEST_PASS_LINE,
        "source": copy.deepcopy(qualified_source),
        "test_source": git_blob_ref(qualified_source, EXECUTOR_PATH),
    }


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


def validate_raw_execution(
    path: Path,
    *,
    check_id: str,
    entrypoint: str,
    profile_detail: str,
    build_source: dict[str, Any],
    binary_sha256: str,
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
    profile_startup_error_count = 0
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
                attributes = event.get("attributes")
                request_id = event.get("request_id")
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
    require(
        reusable_spans > 0 and reusable_fingerprints,
        f"{check_id} did not reach startup reusable executable capture/replay",
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
    }
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


def validate_witness(
    path: Path,
    *,
    check_id: str,
    entrypoint: str,
    profile_detail: str,
    build_source: dict[str, Any],
    binary_sha256: str,
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
    )
    require(
        witness.get("product") == derived["product"]
        and derived["product"].get("success") is True,
        f"{check_id} product summary differs from raw evidence",
    )
    require(
        witness.get("target_signal") == derived["target_signal"]
        and derived["target_signal"]["startup_catalog_error_count"] == 0
        and derived["target_signal"]["startup_reusable_execution_span_count"] > 0
        and derived["target_signal"]["startup_reusable_program_fingerprint_count"] > 0,
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
    return {
        "check_id": check_id,
        "witness": file_ref(path),
        "raw_execution": derived,
    }


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


def plan_classification(
    base_source: dict[str, Any],
    source: dict[str, Any],
    rules_path: Path,
    planner: ModuleType,
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
        if match.get("profile_id") == PROFILE_ID
    ]
    product_changed = [path for path in changed if is_release_product_path(path)]
    require(plan.get("status") == "pass" and not plan.get("unknown_files"), "change-impact plan did not close")
    require(product_changed == [EXECUTOR_PATH], f"qualification contains other product changes: {product_changed}")
    require(len(matches) == 1 and matches[0].get("path") == EXECUTOR_PATH, "qualification profile did not uniquely match executor")
    require(plan.get("required_checks") == EXPECTED_CHECKS, "qualification required check set differs")
    require(plan.get("qualified_scopes") == EXPECTED_SCOPES, "qualification scope set differs")
    require(
        plan.get("required_gates") == EXPECTED_CONTROL_GATES,
        "qualification control gate set is not exactly closed",
    )
    return {
        "profile_id": PROFILE_ID,
        "selector": matches[0],
        "required_checks": EXPECTED_CHECKS,
        "qualified_scopes": EXPECTED_SCOPES,
        "product_changed_files": product_changed,
        "control_plane_changed_files": [
            path for path in changed if path not in product_changed
        ],
        "impact_domains": plan["impact_domains"],
        "required_control_gates": plan["required_gates"],
        "causal_edges": [
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
        ],
    }


def plan_classification_at_source(
    base_source: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    temporary, planner, rules_path = historical_planner(source)
    try:
        return plan_classification(base_source, source, rules_path, planner)
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
    require(docs, "qualification docs review has no changed documents")
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


def revalidated_cells(binary_sha256: str) -> list[dict[str, Any]]:
    return [
        {
            "cell_id": "cuda.run.profile_full",
            "backend": "cuda",
            "entrypoint": "run",
            "profile_detail": "full",
            "evidence": "diagnostic_observability",
            "check_id": "cuda_run_profile_full",
            "binary_sha256": binary_sha256,
        },
        {
            "cell_id": "cuda.serve.profile_verify",
            "backend": "cuda",
            "entrypoint": "serve",
            "profile_detail": "verify",
            "evidence": "diagnostic_observability",
            "check_id": "cuda_serve_profile_verify",
            "binary_sha256": binary_sha256,
        },
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
    unit: dict[str, Any],
    control_gate: dict[str, Any],
    run_full: dict[str, Any],
    serve_verify: dict[str, Any],
) -> dict[str, Any]:
    base_source = prior["source"]
    classification = plan_classification_at_source(base_source, source)
    partition = backend_partition(base_source, source)
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
            "unit_exact": unit,
            "release_validator_selftest": control_gate,
            "docs_review": docs_review_closure(classification, source),
            "cuda_run_profile_full": run_full,
            "cuda_serve_profile_verify": serve_verify,
            "backend_capability_partition": partition,
        },
        "reused_cells": prior["reused_cells"],
        "revalidated_cells": revalidated_cells(candidate["binary_sha256"]),
        "invalidated_cells": [],
        "open_invalidated_cells": [],
        "backend_binary_sha256": authority,
        "prior_reachability": prior["reachability"],
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
    unit_receipt: Path,
    control_gate_receipt: Path,
    run_full_witness: Path,
    serve_verify_witness: Path,
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
    )
    unit = validate_unit_receipt(unit_receipt.expanduser().resolve(), source)
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
    )
    serve_verify = validate_witness(
        serve_verify_witness.expanduser().resolve(),
        check_id="cuda_serve_profile_verify",
        entrypoint="serve",
        profile_detail="verify",
        build_source=candidate["build_source"],
        binary_sha256=candidate["binary_sha256"],
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
            unit=unit,
            control_gate=control_gate,
            run_full=run_full,
            serve_verify=serve_verify,
        )
        write_json(output / "manifest.json", document)
        verify_manifest(output / "manifest.json", verify_checkout=True)
        return str(document["pass_line"])
    except BaseException:
        if output.is_dir() and not output.is_symlink():
            shutil.rmtree(output)
        raise


def verify_manifest(
    manifest_path: Path,
    *,
    verify_checkout: bool = True,
    expected_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = manifest_path.expanduser().resolve()
    root = path.parent
    manifest = read_json(path, "change-impact qualification manifest")
    required_fields = {
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
    require(set(manifest) == required_fields, "qualification manifest field set differs")
    require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("artifact_type") == ARTIFACT_TYPE
        and manifest.get("status") == "pass"
        and manifest.get("profile_id") == PROFILE_ID
        and Path(str(manifest.get("artifact_dir", ""))).resolve() == root
        and manifest.get("does_not_prove") == DOES_NOT_PROVE
        and manifest.get("pass_line") == f"{PASS_PREFIX}: {root}",
        "qualification identity/status/PASS differs",
    )
    source = normalize_source(manifest.get("source"), "qualification")
    expected = current_source() if verify_checkout else expected_source
    if expected is not None:
        require(source == expected, "qualification source is stale")
    prior_path = validate_ref(manifest.get("prior_r1"), "prior R1 manifest")
    prior = prior_r1_summary(prior_path)
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
    classification = plan_classification_at_source(prior["source"], source)
    require(
        manifest.get("classification") == classification,
        "qualification classification differs",
    )
    proofs = manifest.get("proofs")
    require(
        isinstance(proofs, dict)
        and set(proofs)
        == {
            "candidate_cuda",
            "unit_exact",
            "release_validator_selftest",
            "docs_review",
            "cuda_run_profile_full",
            "cuda_serve_profile_verify",
            "backend_capability_partition",
        },
        "qualification proof set differs",
    )
    candidate_recorded = proofs["candidate_cuda"]
    require(isinstance(candidate_recorded, dict), "candidate CUDA proof is invalid")
    binary_path = validate_ref(candidate_recorded.get("binary"), "candidate CUDA binary")
    receipt_path = validate_ref(
        candidate_recorded.get("build_receipt"), "candidate CUDA build receipt"
    )
    candidate = validate_candidate_build(binary_path, receipt_path, source)
    require(candidate_recorded == candidate, "candidate CUDA proof drifted")
    unit_recorded = proofs["unit_exact"]
    require(isinstance(unit_recorded, dict), "unit proof is invalid")
    unit_path = validate_ref(unit_recorded.get("receipt"), "exact unit receipt")
    require(
        unit_recorded == validate_unit_receipt(unit_path, source),
        "unit proof drifted",
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
    require(
        classification["required_control_gates"]
        == sorted([docs_recorded["gate"], *control_recorded["gates"]]),
        "qualification control gates are not fully proved",
    )
    run_recorded = proofs["cuda_run_profile_full"]
    serve_recorded = proofs["cuda_serve_profile_verify"]
    require(isinstance(run_recorded, dict) and isinstance(serve_recorded, dict), "product witness proofs are invalid")
    run_path = validate_ref(run_recorded.get("witness"), "CUDA run/full witness")
    serve_path = validate_ref(serve_recorded.get("witness"), "CUDA serve/verify witness")
    run_full = validate_witness(
        run_path,
        check_id="cuda_run_profile_full",
        entrypoint="run",
        profile_detail="full",
        build_source=candidate["build_source"],
        binary_sha256=candidate["binary_sha256"],
    )
    serve_verify = validate_witness(
        serve_path,
        check_id="cuda_serve_profile_verify",
        entrypoint="serve",
        profile_detail="verify",
        build_source=candidate["build_source"],
        binary_sha256=candidate["binary_sha256"],
    )
    require(run_recorded == run_full, "CUDA run/full witness proof drifted")
    require(serve_recorded == serve_verify, "CUDA serve/verify witness proof drifted")
    partition = backend_partition(prior["source"], source)
    require(
        proofs["backend_capability_partition"] == partition,
        "backend capability partition drifted",
    )
    reused = prior["reused_cells"]
    revalidated = revalidated_cells(candidate["binary_sha256"])
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
    return {
        "kind": "runtime-vnext-change-impact-qualification",
        "manifest": file_ref(path),
        "profile_id": PROFILE_ID,
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
        [row["cell_id"] for row in cells]
        == ["cuda.run.profile_full", "cuda.serve.profile_verify"]
        and all(row["binary_sha256"] == binary for row in cells),
        "self-test revalidated cell set differs",
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
    with tempfile.TemporaryDirectory(
        prefix="ferrum-vnext-impact-witness-selftest-"
    ) as temp_name:
        temp_root = Path(temp_name)
        binary_path = temp_root / "ferrum"
        binary_path.write_bytes(b"self-test CUDA binary")
        raw_binary_sha = sha256(binary_path)
        build_source = {
            "git_sha": "a" * 40,
            "git_tree_sha": "b" * 40,
            "dirty": False,
        }
        build_receipt_path = temp_root / "candidate-build-receipt.json"
        write_json(
            build_receipt_path,
            {
                "schema_version": 1,
                "artifact_type": "runtime_vnext_candidate_build_receipt",
                "status": "pass",
                "backend": "cuda",
                "binary_sha256": raw_binary_sha,
                "source_git_sha": build_source["git_sha"],
                "source_tree_sha": build_source["git_tree_sha"],
            },
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
            profile_path.write_text(
                json.dumps(
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
                    },
                    sort_keys=True,
                )
                + "\n",
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
                    "startup_reusable_execution_span_count": 1,
                    "startup_reusable_program_fingerprint_count": 1,
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
    with tempfile.TemporaryDirectory(
        prefix="ferrum-qualification-closure-selftest-"
    ) as temporary:
        root = Path(temporary)
        source = source_at(git_text("rev-parse", "HEAD"))
        unit_path = root / "unit.receipt.json"
        unit_stdout = root / "unit.stdout.log"
        unit_stderr = root / "unit.stderr.log"
        unit_stdout.write_text(UNIT_TEST_PASS_LINE + "\n", encoding="utf-8")
        unit_stderr.write_text("", encoding="utf-8")
        unit_receipt = {
            "schema": "ferrum.bounded-command-receipt.v1",
            "status": "pass",
            "rc": 0,
            "reason": "command_completed",
            "command": exact_unit_test_command(source),
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
    require(self_test() == 0, "qualification focused self-test failed")

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
        EXACT_UNIT_FILTER,
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
        and f"test {EXACT_UNIT_FILTER} ... ok" in combined
        and "test result: ok. 1 passed; 0 failed" in combined,
        "exact unit cargo output did not prove 1/1 PASS",
    )
    require(current_source() == source, "exact unit source changed during execution")
    print(UNIT_TEST_PASS_LINE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-r1", type=Path)
    parser.add_argument("--profile-id", default=PROFILE_ID)
    parser.add_argument("--candidate-cuda-binary", type=Path)
    parser.add_argument("--candidate-cuda-build-receipt", type=Path)
    parser.add_argument("--unit-receipt", type=Path)
    parser.add_argument("--control-gate-receipt", type=Path)
    parser.add_argument("--run-full-witness", type=Path)
    parser.add_argument("--serve-verify-witness", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--control-self-test", action="store_true")
    parser.add_argument("--run-exact-unit-test", action="store_true")
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
        if args.verify_manifest is not None:
            verified = verify_manifest(args.verify_manifest, verify_checkout=True)
            print(f"{PASS_PREFIX}: {Path(verified['manifest']['path']).parent}")
            return 0
        required = {
            "prior_r1": args.prior_r1,
            "candidate_cuda_binary": args.candidate_cuda_binary,
            "candidate_cuda_build_receipt": args.candidate_cuda_build_receipt,
            "unit_receipt": args.unit_receipt,
            "control_gate_receipt": args.control_gate_receipt,
            "run_full_witness": args.run_full_witness,
            "serve_verify_witness": args.serve_verify_witness,
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
                unit_receipt=args.unit_receipt,
                control_gate_receipt=args.control_gate_receipt,
                run_full_witness=args.run_full_witness,
                serve_verify_witness=args.serve_verify_witness,
                out=args.out,
            )
        )
        return 0
    except (OSError, QualificationError, RuntimeError, ValueError) as error:
        print(f"FERRUM RUNTIME VNEXT CHANGE IMPACT QUALIFICATION FAIL: {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
