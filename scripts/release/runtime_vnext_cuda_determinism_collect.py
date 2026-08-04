#!/usr/bin/env python3
"""Collect and assemble auditable CUDA vNext determinism evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import runtime_vnext_cuda_determinism as validator
import runtime_vnext_baseline_scenarios as baseline_scenarios


REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_PATH = Path(__file__).resolve()
COLLECTOR_RELATIVE_PATH = COLLECTOR_PATH.relative_to(REPO_ROOT).as_posix()
BOUNDED_COMMAND_PATH = REPO_ROOT / "scripts/release/bounded_command.py"
HARDWARE_PROBE_PATH = REPO_ROOT / "scripts/release/runtime_vnext_hardware_probe.py"
COLLECTOR_ARTIFACT_TYPE = "runtime_vnext_cuda_determinism_collector"
MODEL_VERIFICATION_ARTIFACT_TYPE = (
    "runtime_vnext_cuda_determinism_model_verification"
)
EVIDENCE_READY_PREFIX = "FERRUM RUNTIME VNEXT CUDA DETERMINISM EVIDENCE READY"
SELFTEST_PASS_LINE = "RUNTIME VNEXT CUDA DETERMINISM COLLECTOR SELF-TEST PASS"
MAX_PROCESSES = 16
MAX_GROUP_THREADS = 128
MAX_PER_PROCESS_THREADS = 64
MAX_WALL_SECONDS = 6 * 60 * 60
MAX_PREFLIGHT_SECONDS = 60 * 60
MAX_BUILD_PROVENANCE_BYTES = 2 * 1024 * 1024 * 1024
TOKENIZER_REQUIRED_FILES = ("tokenizer.json",)
TOKENIZER_OPTIONAL_FILES = (
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "chat_template.jinja",
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


class CollectionError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectionError(message)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_file_sha256(path: Path, *, deadline: float, label: str) -> str:
    digest = hashlib.sha256()
    completed = 0
    next_byte_progress = 1024**3
    next_time_progress = time.monotonic() + 10.0
    with path.open("rb") as handle:
        while True:
            require(time.monotonic() < deadline, f"model verification deadline exceeded: {path}")
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            completed += len(chunk)
            now = time.monotonic()
            if completed >= next_byte_progress or now >= next_time_progress:
                print(
                    "FERRUM VNEXT DETERMINISM MODEL VERIFY PROGRESS "
                    f"model_file={label} bytes={completed}",
                    flush=True,
                )
                next_byte_progress = completed + 1024**3
                next_time_progress = now + 10.0
    return digest.hexdigest()


def file_ref(
    root: Path,
    path: Path,
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise CollectionError(f"artifact file is outside root: {resolved_path}") from error
    require(path.is_file() and not path.is_symlink(), f"artifact file is not regular: {path}")
    size = path.stat().st_size
    require(allow_empty or size > 0, f"artifact file is empty: {path}")
    return {"path": relative, "sha256": file_sha256(path), "size_bytes": size}


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CollectionError(f"invalid JSON {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def exact_object(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    require(actual == fields, f"{label} fields differ: {sorted(actual ^ fields)}")
    return value


def sanitized_environment() -> dict[str, str]:
    try:
        return validator.canonical_determinism_environment(os.environ)
    except validator.DeterminismArtifactError as error:
        raise CollectionError(f"cannot construct the fixed runner environment: {error}") from error


def command_output(*argv: str) -> str:
    result = subprocess.run(
        list(argv),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        result.returncode == 0,
        f"command failed {argv!r}: {result.stderr.strip()}",
    )
    return result.stdout.strip()


def source_identity() -> dict[str, str]:
    git_sha = command_output("git", "rev-parse", "HEAD")
    git_tree_sha = command_output("git", "rev-parse", "HEAD^{tree}")
    dirty = command_output("git", "status", "--short", "--untracked-files=all")
    require(not dirty, "tracked worktree must be clean before CUDA collection")
    require(
        validator.GIT_SHA_RE.fullmatch(git_sha) is not None
        and validator.GIT_SHA_RE.fullmatch(git_tree_sha) is not None,
        "source git identity is invalid",
    )
    return {"git_sha": git_sha, "git_tree_sha": git_tree_sha}


def prepare_empty_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if root.exists():
        require(root.is_dir() and not root.is_symlink(), "artifact root is not a real directory")
        require(not any(root.iterdir()), "artifact root must be empty")
    else:
        root.mkdir(parents=True)
    return root


def copy_exclusive(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    require(source.is_file() and not source.is_symlink(), f"input is not a real file: {source}")
    require(not destination.exists(), f"artifact destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    require(
        destination.is_file()
        and not destination.is_symlink()
        and file_sha256(destination) == file_sha256(source),
        f"copied artifact differs from source: {source}",
    )


def import_candidate_build_provenance(
    root: Path,
    candidate_root: Path,
    *,
    source: dict[str, str],
    hardware_id: str,
    allow_internal_fixture: bool = False,
) -> None:
    root = root.resolve()
    candidate_root = candidate_root.expanduser().resolve()
    require(
        candidate_root.is_dir() and not candidate_root.is_symlink(),
        f"candidate build root is not a real directory: {candidate_root}",
    )
    receipt_path = candidate_root / "build/candidate/candidate-build-receipt.json"
    require(
        receipt_path.is_file() and not receipt_path.is_symlink(),
        f"candidate build receipt is missing: {receipt_path}",
    )
    receipt = read_json(receipt_path)
    binary_sha = receipt.get("binary_sha256")
    require(
        isinstance(binary_sha, str)
        and validator.SHA256_RE.fullmatch(binary_sha) is not None,
        "candidate build receipt binary SHA256 is invalid",
    )
    expected = {
        "source_git_sha": source["git_sha"],
        "source_tree_sha": source["git_tree_sha"],
        "hardware_id": hardware_id,
        "backend": "cuda",
        "binary_sha256": binary_sha,
    }
    try:
        _, _, _, source_binary = baseline_scenarios.validate_candidate_build_receipt(
            candidate_root,
            baseline_scenarios.existing_artifact_ref(
                candidate_root, receipt_path, "raw-json"
            ),
            expected=expected,
            allow_internal_fixture=allow_internal_fixture,
        )
    except baseline_scenarios.ScenarioError as error:
        raise CollectionError(f"candidate build receipt is invalid: {error}") from error
    require(
        source_binary == candidate_root / "build/candidate/ferrum",
        "candidate build receipt does not bind build/candidate/ferrum",
    )

    recorded_lock = receipt.get("native_operator_set_lock")
    require(isinstance(recorded_lock, dict), "CUDA candidate receipt lacks native operator lock")
    source_lock = (
        candidate_root / baseline_scenarios.CANDIDATE_NATIVE_OPERATOR_SET_LOCK_REL
    )
    require(
        source_lock.is_file() and not source_lock.is_symlink(),
        "candidate native operator set lock is absent from the portable build artifact",
    )
    lock_stat = source_lock.stat()
    require(
        file_sha256(source_lock) == recorded_lock.get("sha256")
        and lock_stat.st_size == recorded_lock.get("size_bytes"),
        "portable candidate native operator set lock differs from the release build",
    )

    source_tree = candidate_root / "build/candidate"
    total_bytes = 0
    for path in source_tree.rglob("*"):
        require(not path.is_symlink(), f"candidate build provenance contains a symlink: {path}")
        if path.is_file():
            total_bytes += path.stat().st_size
            require(
                total_bytes <= MAX_BUILD_PROVENANCE_BYTES,
                "candidate build provenance exceeds the 2 GiB copy bound",
            )
    provenance_root = root / validator.BUILD_PROVENANCE_ROOT
    imported_tree = provenance_root / "build/candidate"
    imported_tree.parent.mkdir(parents=True)
    shutil.copytree(source_tree, imported_tree, symlinks=False)
    imported_receipt = provenance_root / "build/candidate/candidate-build-receipt.json"
    try:
        _, _, _, imported_binary = baseline_scenarios.validate_candidate_build_receipt(
            provenance_root,
            baseline_scenarios.existing_artifact_ref(
                provenance_root, imported_receipt, "raw-json"
            ),
            expected=expected,
            allow_internal_fixture=allow_internal_fixture,
        )
    except baseline_scenarios.ScenarioError as error:
        raise CollectionError(
            f"imported candidate build provenance is invalid: {error}"
        ) from error
    require(
        imported_binary == root / validator.CANDIDATE_BUILD_BINARY_PATH,
        "imported candidate binary is outside its canonical layout",
    )
    imported_lock = root / validator.NATIVE_OPERATOR_SET_LOCK_PATH
    require(
        imported_lock.is_file()
        and not imported_lock.is_symlink()
        and file_sha256(imported_lock) == recorded_lock.get("sha256")
        and imported_lock.stat().st_size == recorded_lock.get("size_bytes"),
        "imported candidate native operator set lock differs from the release build",
    )
    copy_exclusive(imported_binary, root / "binary/ferrum")
    require(
        file_sha256(root / "binary/ferrum") == binary_sha,
        "canonical determinism binary differs from the candidate build receipt",
    )


def parse_model_bindings(values: list[str], scope: str) -> dict[str, Path]:
    expected = set(validator.scope_contract(scope)["models"])
    bindings: dict[str, Path] = {}
    for raw in values:
        key, separator, directory = raw.partition("=")
        require(separator == "=" and key and directory, f"invalid --model binding: {raw}")
        require(key in expected and key not in bindings, f"invalid or duplicate model key: {key}")
        resolved = Path(directory).expanduser().resolve()
        require(resolved.is_dir(), f"model directory is missing: {resolved}")
        bindings[key] = resolved
    require(set(bindings) == expected, "--model bindings differ from the selected scope")
    return {key: bindings[key] for key in sorted(bindings)}


def safe_model_file(model_dir: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    require(
        not pure.is_absolute()
        and pure.parts
        and ".." not in pure.parts
        and pure.as_posix() == relative,
        f"model lock contains an unsafe file path: {relative}",
    )
    path = model_dir.joinpath(*pure.parts)
    require(path.is_file(), f"locked model file is missing: {path}")
    return path


def valid_relative_safetensors_path(relative: str) -> bool:
    pure = PurePosixPath(relative)
    return (
        bool(relative)
        and not pure.is_absolute()
        and pure.as_posix() == relative
        and all(component not in {"", ".", ".."} for component in pure.parts)
        and relative.endswith(".safetensors")
    )


def production_consumed_model_files(model_dir: Path) -> list[str]:
    consumed = {"config.json", *TOKENIZER_REQUIRED_FILES}
    for relative in consumed:
        require(
            model_dir.joinpath(*PurePosixPath(relative).parts).is_file(),
            f"required production model file is missing: {model_dir / relative}",
        )
    consumed.update(
        relative
        for relative in TOKENIZER_OPTIONAL_FILES
        if model_dir.joinpath(*PurePosixPath(relative).parts).is_file()
    )

    top_level_safetensors = sorted(
        path.name
        for path in model_dir.iterdir()
        if path.is_file() and path.name.endswith(".safetensors")
    )
    require(top_level_safetensors, f"safetensors source contains no shard: {model_dir}")
    consumed.update(top_level_safetensors)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        consumed.add("model.safetensors.index.json")

    if not (model_dir / "model.safetensors").is_file():
        require(index_path.is_file(), f"sharded safetensors source lacks its index: {model_dir}")
        require(
            index_path.stat().st_size <= validator.MAX_MODEL_LOCK_JSON_BYTES,
            "safetensors index exceeds the 64 MiB preflight bound",
        )
        index = read_json(index_path)
        weight_map = index.get("weight_map")
        require(isinstance(weight_map, dict) and weight_map, "safetensors index lacks weight_map")
        for tensor_name, raw_relative in weight_map.items():
            require(
                isinstance(tensor_name, str)
                and tensor_name
                and isinstance(raw_relative, str)
                and valid_relative_safetensors_path(raw_relative),
                "safetensors index contains an invalid shard path",
            )
            safe_model_file(model_dir, raw_relative)
            consumed.add(raw_relative)
    return sorted(consumed)


def require_consumed_files_locked(
    model_key: str,
    model_dir: Path,
    locked_paths: set[str],
) -> list[str]:
    consumed_files = production_consumed_model_files(model_dir)
    missing_consumed = sorted(set(consumed_files) - locked_paths)
    require(
        not missing_consumed,
        f"production-consumed files are absent from models.lock for {model_key}: {missing_consumed}",
    )
    return consumed_files


def verify_model_files(
    bindings: dict[str, Path],
    locked_models: dict[str, dict[str, Any]],
    *,
    deadline: float,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], tuple[int, int, int, int]]]:
    models = []
    identities: dict[tuple[str, str], tuple[int, int, int, int]] = {}
    for model_key, model_dir in bindings.items():
        locked_paths = {item["path"] for item in locked_models[model_key]["files"]}
        consumed_files = require_consumed_files_locked(
            model_key,
            model_dir,
            locked_paths,
        )
        files = []
        for locked in locked_models[model_key]["files"]:
            path = safe_model_file(model_dir, locked["path"])
            stat = path.stat()
            require(
                stat.st_size == locked["size_bytes"],
                f"locked model file size differs: {path}",
            )
            print(
                "FERRUM VNEXT DETERMINISM MODEL VERIFY "
                f"model={model_key} file={locked['path']} bytes={stat.st_size}",
                flush=True,
            )
            digest = model_file_sha256(
                path,
                deadline=deadline,
                label=f"{model_key}/{locked['path']}",
            )
            require(digest == locked["sha256"], f"locked model file SHA256 differs: {path}")
            files.append(dict(locked))
            identities[(model_key, locked["path"])] = (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
            )
        models.append(
            {
                "model_key": model_key,
                "model_dir": str(model_dir),
                "files": files,
                "consumed_files": consumed_files,
            }
        )
    return models, identities


def require_models_unchanged(
    bindings: dict[str, Path],
    identities: dict[tuple[str, str], tuple[int, int, int, int]],
) -> None:
    for (model_key, relative), expected in identities.items():
        path = safe_model_file(bindings[model_key], relative)
        stat = path.stat()
        actual = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        require(actual == expected, f"model file changed during collection: {path}")


def run_hardware_probe(
    root: Path,
    hardware_id: str,
    environment: dict[str, str],
    *,
    deadline: float,
) -> None:
    output = root / "hardware-probe"
    logs = root / "preflight"
    logs.mkdir()
    remaining = deadline - time.monotonic()
    require(remaining > 0, "preflight deadline expired before hardware probe")
    result = subprocess.run(
        [
            sys.executable,
            str(HARDWARE_PROBE_PATH),
            "--backend",
            "cuda",
            "--hardware-id",
            hardware_id,
            "--policy-id",
            "cuda-g0-1x-rtx4090",
            "--out",
            str(output),
            "--source-root",
            str(REPO_ROOT),
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=min(120.0, remaining),
        check=False,
    )
    (logs / "hardware-probe.stdout.log").write_text(result.stdout, encoding="utf-8")
    (logs / "hardware-probe.stderr.log").write_text(result.stderr, encoding="utf-8")
    require(result.returncode == 0, f"hardware probe failed: {result.stderr.strip()}")
    expected = f"RUNTIME VNEXT HARDWARE PROBE PASS: {output}"
    require(expected in result.stdout.splitlines(), "hardware probe did not print its exact PASS line")


def collector_command(
    root: Path,
    binary: Path,
    bindings: dict[str, Path],
    scope: str,
) -> list[str]:
    command = [
        str(binary),
        "vnext-determinism",
        "--models-lock",
        str(root / "models.lock.json"),
        "--artifact-root",
        str(root),
    ]
    if scope == validator.M1_S2_FOCUSED_SCOPE:
        command.extend(["--scope", scope])
    for model_key, directory in bindings.items():
        command.extend(["--model", f"{model_key}={directory}"])
    return command


def run_bounded_collector(
    root: Path,
    command: list[str],
    wall_timeout_seconds: int,
    environment: dict[str, str],
) -> dict[str, Any]:
    require(
        1 <= wall_timeout_seconds <= MAX_WALL_SECONDS,
        f"wall timeout must be in [1, {MAX_WALL_SECONDS}] seconds",
    )
    runner = root / "runner"
    runner.mkdir()
    receipt = runner / "receipt.json"
    stdout = runner / "stdout.log"
    stderr = runner / "stderr.log"
    wrapper = [
        sys.executable,
        str(BOUNDED_COMMAND_PATH),
        "--receipt",
        str(receipt),
        "--stdout-log",
        str(stdout),
        "--stderr-log",
        str(stderr),
        "--cwd",
        str(REPO_ROOT),
        "--wall-timeout-seconds",
        str(wall_timeout_seconds),
        "--max-processes",
        str(MAX_PROCESSES),
        "--max-group-threads",
        str(MAX_GROUP_THREADS),
        "--max-per-process-threads",
        str(MAX_PER_PROCESS_THREADS),
        "--sample-interval-seconds",
        "0.2",
        "--max-sampling-errors",
        "3",
        "--term-grace-seconds",
        "10",
        "--",
        *command,
    ]
    result = subprocess.run(
        wrapper,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=wall_timeout_seconds + 60,
        check=False,
    )
    print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    require(receipt.is_file(), "bounded collector did not produce a receipt")
    receipt_value = read_json(receipt)
    require(
        result.returncode == 0
        and receipt_value.get("status") == "pass"
        and receipt_value.get("reason") == "command_completed"
        and receipt_value.get("rc") == 0,
        f"bounded collector failed; inspect {runner}",
    )
    return receipt_value


def collector_pass_line(scope: str, artifact_root: str) -> str:
    prefix = (
        "FERRUM VNEXT M1 S2 FOCUSED DETERMINISM COLLECTOR PASS"
        if scope == validator.M1_S2_FOCUSED_SCOPE
        else "FERRUM VNEXT DETERMINISM COLLECTOR PASS"
    )
    return f"{prefix}: {artifact_root}"


def validate_collector(
    root: Path,
    command: list[str],
    bindings: dict[str, Path],
    scope: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    collector_path = root / "collector.json"
    require(collector_path.is_file(), "collector.json is missing")
    collector = exact_object(read_json(collector_path), COLLECTOR_FIELDS, "collector")
    require(
        collector["schema_version"] == 1
        and collector["artifact_type"] == COLLECTOR_ARTIFACT_TYPE
        and collector["status"] == "pass"
        and collector["backend"] == "cuda"
        and collector["scope"] == scope,
        "collector identity or status is invalid",
    )
    expected_pass = collector_pass_line(scope, command[5])
    require(collector["pass_line"] == expected_pass, "collector PASS line is invalid")
    stdout_lines = (root / "runner/stdout.log").read_text(encoding="utf-8").splitlines()
    require(expected_pass in stdout_lines, "collector stdout lacks the exact PASS line")

    lock_ref = file_ref(root, root / "models.lock.json")
    probe_ref = file_ref(root, root / "hardware-probe/probe.json")
    require(collector["models_lock"] == lock_ref, "collector model lock reference differs")
    require(collector["hardware_probe"] == probe_ref, "collector hardware reference differs")
    probe = read_json(root / "hardware-probe/probe.json")
    require(
        collector["device_fingerprint"] == probe.get("fingerprint"),
        "collector device fingerprint differs from hardware probe",
    )

    binary = root / "binary/ferrum"
    expected_binary = {
        "path": str(binary.resolve()),
        "sha256": file_sha256(binary),
        "size_bytes": binary.stat().st_size,
    }
    require(collector["binary"] == expected_binary, "collector binary identity differs")
    validator.validate_file_ref(
        root,
        {
            key: collector["denominator"][key]
            for key in validator.FILE_REF_FIELDS
        },
        "collector.denominator",
    )

    expected_models = set(validator.scope_contract(scope)["models"])
    expected_case_count = (
        len(expected_models)
        * len(validator.scope_contract(scope)["partitions"])
        * 4
    )
    require(
        collector["case_count"] == expected_case_count
        and len(collector["cases"]) == expected_case_count,
        "collector case denominator differs from the selected scope",
    )
    for index, case_ref in enumerate(collector["cases"]):
        validator.validate_file_ref(root, case_ref, f"collector.cases[{index}]")
    require(
        [row["path"] for row in collector["cases"]]
        == sorted(row["path"] for row in collector["cases"]),
        "collector cases are not canonical",
    )

    model_rows = collector["models"]
    require(isinstance(model_rows, list), "collector.models must be a list")
    model_index: dict[str, dict[str, Any]] = {}
    cases_per_model = len(validator.scope_contract(scope)["partitions"]) * 4
    for index, raw in enumerate(model_rows):
        row = exact_object(raw, COLLECTOR_MODEL_FIELDS, f"collector.models[{index}]")
        key = row["model_key"]
        require(key in expected_models and key not in model_index, "collector model is invalid")
        require(
            row["model_dir"] == str(bindings[key])
            and row["case_count"] == cases_per_model,
            f"collector model binding or case count differs for {key}",
        )
        validator.sha256_text(
            row["resolved_plan_fingerprint"],
            f"collector.models[{index}].resolved_plan_fingerprint",
        )
        validator.sha256_text(row["plan_hash"], f"collector.models[{index}].plan_hash")
        model_index[key] = row
    require(list(model_index) == sorted(expected_models), "collector model set is not canonical")

    denominator = read_json(root / collector["denominator"]["path"])
    normalized_denominator = validator.validate_denominator(
        denominator, expected_models, scope
    )
    for key, row in model_index.items():
        denominator_model = normalized_denominator["models"][key]
        require(
            row["resolved_plan_fingerprint"]
            == denominator_model["resolved_plan_fingerprint"]
            and row["plan_hash"] == denominator_model["plan_hash"],
            f"collector model identity differs from denominator for {key}",
        )
    return collector, normalized_denominator


def assemble_evidence(
    root: Path,
    *,
    scope: str,
    source: dict[str, str],
    command: list[str],
    bindings: dict[str, Path],
    environment: dict[str, str],
    repository_root: Path,
    allow_internal_fixture: bool = False,
) -> dict[str, Any]:
    collector, denominator = validate_collector(root, command, bindings, scope)
    lock_ref = file_ref(root, root / "models.lock.json")
    locked_models = validator.validate_models_lock(root, lock_ref)
    verification_ref = file_ref(root, root / "model-verification.json")
    receipt_path = root / "runner/receipt.json"
    receipt = read_json(receipt_path)
    require(receipt.get("command") == command, "bounded receipt command differs")
    require(receipt.get("rc") == 0, "bounded receipt exit code is not zero")

    models = []
    collector_models = {row["model_key"]: row for row in collector["models"]}
    for key in sorted(bindings):
        locked = locked_models[key]
        planned = denominator["models"][key]
        collected = collector_models[key]
        models.append(
            {
                **locked,
                "external_metadata_id": planned["external_metadata_id"],
                "resolved_plan_fingerprint": collected[
                    "resolved_plan_fingerprint"
                ],
                "plan_hash": collected["plan_hash"],
            }
        )

    probe_path = root / "hardware-probe/probe.json"
    probe = read_json(probe_path)
    runner_environment = validator.validate_runner_environment(environment)
    repository_root = repository_root.resolve()
    evidence = {
        "schema_version": 1,
        "artifact_type": validator.ARTIFACT_TYPE,
        "backend": "cuda",
        "scope": scope,
        "source": {
            "git_sha": source["git_sha"],
            "git_tree_sha": source["git_tree_sha"],
            "dirty_status": [],
            "binary_path": command[0],
            "binary": file_ref(root, root / "binary/ferrum"),
            "candidate_build_receipt": file_ref(
                root,
                root / validator.CANDIDATE_BUILD_RECEIPT_PATH,
            ),
            "native_operator_set_lock": file_ref(
                root,
                root / validator.NATIVE_OPERATOR_SET_LOCK_PATH,
            ),
        },
        "hardware": {
            "probe": file_ref(root, probe_path),
            "fingerprint": probe["fingerprint"],
        },
        "models_lock": lock_ref,
        "model_verification": verification_ref,
        "collector": file_ref(root, root / "collector.json"),
        "denominator": collector["denominator"],
        "models": models,
        "runner": {
            "command": command,
            "environment": runner_environment,
            "repository_root": str(repository_root),
            "started_at": receipt["started_at"],
            "finished_at": receipt["ended_at"],
            "exit_code": receipt["rc"],
            "receipt": file_ref(root, receipt_path),
            "stdout": file_ref(root, root / "runner/stdout.log"),
            "stderr": file_ref(
                root,
                root / "runner/stderr.log",
                allow_empty=True,
            ),
        },
        "cases": collector["cases"],
    }
    write_json(root / "evidence.json", evidence, exclusive=True)
    try:
        return validator.validate_artifact(
            root,
            source,
            scope,
            allow_internal_fixture=allow_internal_fixture,
        )
    except validator.DeterminismArtifactError as error:
        raise CollectionError(f"assembled evidence rejected by validator: {error}") from error


def collect(args: argparse.Namespace) -> int:
    scope = args.scope
    root = prepare_empty_root(args.artifact_root)
    source = source_identity()
    environment = sanitized_environment()
    bindings = parse_model_bindings(args.model, scope)
    require(
        1 <= args.preflight_timeout_seconds <= MAX_PREFLIGHT_SECONDS,
        f"preflight timeout must be in [1, {MAX_PREFLIGHT_SECONDS}] seconds",
    )
    preflight_deadline = time.monotonic() + args.preflight_timeout_seconds
    print(
        "FERRUM VNEXT DETERMINISM PREFLIGHT START "
        f"scope={scope} deadline_seconds={args.preflight_timeout_seconds} "
        "progress=model-file-bytes,hardware-probe",
        flush=True,
    )

    copy_exclusive(args.models_lock, root / "models.lock.json")
    lock_ref = file_ref(root, root / "models.lock.json")
    locked_models = validator.validate_models_lock(root, lock_ref)
    verified_models, model_identities = verify_model_files(
        bindings,
        locked_models,
        deadline=preflight_deadline,
    )
    write_json(
        root / "model-verification.json",
        {
            "schema_version": 1,
            "artifact_type": MODEL_VERIFICATION_ARTIFACT_TYPE,
            "scope": scope,
            "source_git_sha": source["git_sha"],
            "source_tree_sha": source["git_tree_sha"],
            "collector": {
                "path": COLLECTOR_RELATIVE_PATH,
                "sha256": file_sha256(COLLECTOR_PATH),
            },
            "verified_at": now_iso(),
            "models": verified_models,
        },
        exclusive=True,
    )

    import_candidate_build_provenance(
        root,
        args.candidate_build_root,
        source=source,
        hardware_id=args.hardware_id,
    )
    binary = root / "binary/ferrum"
    require(binary.stat().st_mode & 0o111 != 0, "imported ferrum binary is not executable")
    run_hardware_probe(
        root,
        args.hardware_id,
        environment,
        deadline=preflight_deadline,
    )
    command = collector_command(root, binary, bindings, scope)
    print(
        "FERRUM VNEXT DETERMINISM START "
        f"scope={scope} deadline_seconds={args.wall_timeout_seconds} "
        f"progress_dir={root / 'collector-progress/cases'} "
        "progress_stdout='FERRUM VNEXT DETERMINISM PROGRESS complete=x/y'",
        flush=True,
    )
    run_bounded_collector(
        root,
        command,
        args.wall_timeout_seconds,
        environment,
    )
    require_models_unchanged(bindings, model_identities)
    require(source_identity() == source, "source identity changed during collection")
    summary = assemble_evidence(
        root,
        scope=scope,
        source=source,
        command=command,
        bindings=bindings,
        environment=environment,
        repository_root=REPO_ROOT,
    )
    print(f"{EVIDENCE_READY_PREFIX}: {root}")
    print(json.dumps(summary, sort_keys=True))
    return 0


def prepare_selftest_fixture(
    root: Path,
    source: dict[str, str],
    scope: str,
) -> tuple[list[str], dict[str, Path]]:
    validator.make_selftest_artifact(root, source, scope)
    evidence = read_json(root / "evidence.json")
    command = list(evidence["runner"]["command"])
    command[0] = str((root / "binary/ferrum").resolve())
    command[3] = str((root / "models.lock.json").resolve())
    command[5] = str(root.resolve())
    bindings = {
        row["model_key"]: Path(row["model_dir"])
        for row in read_json(root / "model-verification.json")["models"]
    }
    offset = 8 if scope == validator.M1_S2_FOCUSED_SCOPE else 6
    for index, key in enumerate(sorted(bindings)):
        command[offset + index * 2 + 1] = f"{key}={bindings[key]}"

    stdout_path = root / "runner/stdout.log"
    stdout_path.write_text(
        collector_pass_line(scope, command[5]) + "\n", encoding="utf-8"
    )
    receipt_path = root / "runner/receipt.json"
    receipt = read_json(receipt_path)
    receipt["command"] = command
    receipt["cwd"] = str(REPO_ROOT.resolve())
    receipt["stdout"]["path"] = str((root / "runner/stdout.log").resolve())
    receipt["stderr"]["path"] = str((root / "runner/stderr.log").resolve())
    receipt["stdout"]["sha256"] = file_sha256(stdout_path)
    receipt["stdout"]["size_bytes"] = stdout_path.stat().st_size
    write_json(receipt_path, receipt)

    collector_models = []
    case_count_by_model = {key: 0 for key in bindings}
    case_dtype: dict[str, str] = {}
    case_quantization: dict[str, str] = {}
    for case_ref in evidence["cases"]:
        case = read_json(root / case_ref["path"])
        key = case["model_key"]
        case_count_by_model[key] += 1
        case_dtype[key] = case["dtype"]
        case_quantization[key] = case["quantization"]
    evidence_models = {row["model_key"]: row for row in evidence["models"]}
    for key in sorted(bindings):
        row = evidence_models[key]
        collector_models.append(
            {
                "model_key": key,
                "model_dir": str(bindings[key]),
                "resolved_plan_fingerprint": row["resolved_plan_fingerprint"],
                "plan_hash": row["plan_hash"],
                "dtype": case_dtype[key],
                "quantization": case_quantization[key],
                "case_count": case_count_by_model[key],
            }
        )
    collector = {
        "schema_version": 1,
        "artifact_type": COLLECTOR_ARTIFACT_TYPE,
        "status": "pass",
        "backend": "cuda",
        "scope": scope,
        "models_lock": file_ref(root, root / "models.lock.json"),
        "hardware_probe": file_ref(root, root / "hardware-probe/probe.json"),
        "device_fingerprint": evidence["hardware"]["fingerprint"],
        "binary": {
            "path": command[0],
            "sha256": evidence["source"]["binary"]["sha256"],
            "size_bytes": evidence["source"]["binary"]["size_bytes"],
        },
        "denominator": evidence["denominator"],
        "models": collector_models,
        "cases": evidence["cases"],
        "case_count": len(evidence["cases"]),
        "execution_count": len(evidence["cases"]) * 12,
        "comparison_count": len(evidence["cases"]) * 15,
        "pass_line": collector_pass_line(scope, command[5]),
    }
    write_json(root / "collector.json", collector)
    (root / "evidence.json").unlink()
    return command, bindings


def self_test() -> int:
    source = {
        "git_sha": validator.git_stdout("rev-parse", "HEAD"),
        "git_tree_sha": validator.git_stdout("rev-parse", "HEAD^{tree}"),
    }
    with tempfile.TemporaryDirectory(
        prefix="ferrum-vnext-cuda-determinism-collect-"
    ) as temporary:
        hash_fixture = Path(temporary) / "hash-fixture.bin"
        hash_fixture.write_bytes(b"bounded-model-file")
        require(
            model_file_sha256(
                hash_fixture,
                deadline=time.monotonic() + 1.0,
                label="selftest/hash-fixture.bin",
            )
            == file_sha256(hash_fixture),
            "bounded model hash differs from the canonical digest",
        )
        try:
            model_file_sha256(
                hash_fixture,
                deadline=time.monotonic() - 1.0,
                label="selftest/expired.bin",
            )
        except CollectionError as error:
            require("deadline exceeded" in str(error), "hash deadline rejected ambiguously")
        else:
            raise CollectionError("expired model hash deadline unexpectedly passed")

        closure_root = Path(temporary) / "model-closure"
        closure_root.mkdir()
        for name in ("config.json", "tokenizer.json", "model.safetensors"):
            (closure_root / name).write_text(f"fixture {name}\n", encoding="utf-8")
        base_locked = {"config.json", "tokenizer.json", "model.safetensors"}
        require(
            require_consumed_files_locked(
                "m1-qwen35-4b", closure_root, base_locked
            )
            == sorted(base_locked),
            "base production-consumed closure drifted",
        )
        for extra in ("special_tokens_map.json", "rogue.safetensors"):
            extra_path = closure_root / extra
            extra_path.write_text("unlocked production input\n", encoding="utf-8")
            try:
                require_consumed_files_locked(
                    "m1-qwen35-4b", closure_root, base_locked
                )
            except CollectionError as error:
                require(extra in str(error), f"unlocked {extra} rejection is ambiguous")
            else:
                raise CollectionError(f"unlocked {extra} unexpectedly passed closure preflight")
            extra_path.unlink()

        sharded_root = Path(temporary) / "sharded-closure"
        (sharded_root / "nested").mkdir(parents=True)
        for name in ("config.json", "tokenizer.json", "part-00001.safetensors"):
            (sharded_root / name).write_text(f"fixture {name}\n", encoding="utf-8")
        (sharded_root / "nested/rogue.safetensors").write_text(
            "unlocked nested shard\n", encoding="utf-8"
        )
        write_json(
            sharded_root / "model.safetensors.index.json",
            {
                "weight_map": {
                    "layer.0.weight": "part-00001.safetensors",
                    "layer.1.weight": "nested/rogue.safetensors",
                }
            },
        )
        sharded_locked = {
            "config.json",
            "tokenizer.json",
            "part-00001.safetensors",
            "model.safetensors.index.json",
        }
        try:
            require_consumed_files_locked(
                "m1-qwen35-4b", sharded_root, sharded_locked
            )
        except CollectionError as error:
            require(
                "nested/rogue.safetensors" in str(error),
                "unlocked nested shard rejection is ambiguous",
            )
        else:
            raise CollectionError("unlocked nested shard unexpectedly passed closure preflight")

        for scope, expected_cases in (
            (validator.RELEASE_SCOPE, 72),
            (validator.M1_S2_FOCUSED_SCOPE, 20),
        ):
            root = Path(temporary) / scope
            root.mkdir()
            command, bindings = prepare_selftest_fixture(root, source, scope)
            environment = validator.canonical_determinism_environment(
                {"HOME": "/workspace", "PATH": "/usr/local/bin:/usr/bin:/bin"}
            )
            summary = assemble_evidence(
                root,
                scope=scope,
                source=source,
                command=command,
                bindings=bindings,
                environment=environment,
                repository_root=REPO_ROOT,
                allow_internal_fixture=True,
            )
            require(summary["case_count"] == expected_cases, "self-test case count drifted")
            if scope == validator.M1_S2_FOCUSED_SCOPE:
                try:
                    validator.validate_artifact(root, source, validator.RELEASE_SCOPE)
                except validator.DeterminismArtifactError as error:
                    require("scope differs" in str(error), "focused/full rejection is ambiguous")
                else:
                    raise CollectionError("focused artifact passed the release-full validator")

                candidate_origin = root / validator.BUILD_PROVENANCE_ROOT
                relocated_candidate = Path(temporary) / "candidate-relocated"
                shutil.copytree(candidate_origin, relocated_candidate)
                shutil.rmtree(candidate_origin)
                imported_root = Path(temporary) / "candidate-import"
                imported_root.mkdir()
                import_candidate_build_provenance(
                    imported_root,
                    relocated_candidate,
                    source=source,
                    hardware_id="selftest-cuda-rtx4090",
                    allow_internal_fixture=True,
                )
                require(
                    (imported_root / "binary/ferrum").is_file()
                    and (
                        imported_root / validator.CANDIDATE_BUILD_RECEIPT_PATH
                    ).is_file()
                    and (
                        imported_root / validator.NATIVE_OPERATOR_SET_LOCK_PATH
                    ).is_file(),
                    "candidate build provenance import is incomplete",
                )
    print(SELFTEST_PASS_LINE)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--candidate-build-root", type=Path)
    parser.add_argument("--models-lock", type=Path)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument(
        "--scope",
        choices=sorted(validator.SCOPE_CONTRACTS),
        default=validator.RELEASE_SCOPE,
    )
    parser.add_argument("--hardware-id")
    parser.add_argument("--wall-timeout-seconds", type=int, default=3600)
    parser.add_argument("--preflight-timeout-seconds", type=int, default=1800)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if not all(
        (
            args.artifact_root,
            args.candidate_build_root,
            args.models_lock,
            args.hardware_id,
        )
    ):
        raise SystemExit(
            "--artifact-root, --candidate-build-root, --models-lock, --hardware-id and --model are required"
        )
    require(args.model, "at least one --model binding is required")
    try:
        return collect(args)
    except (
        CollectionError,
        validator.DeterminismArtifactError,
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        if args.artifact_root is not None:
            root = args.artifact_root.expanduser().resolve()
            try:
                if root.is_dir() and not (root / "evidence.json").exists():
                    write_json(
                        root / "collection.reject.json",
                        {
                            "schema_version": 1,
                            "artifact_type": "runtime_vnext_cuda_determinism_collection",
                            "status": "reject",
                            "scope": args.scope,
                            "message": str(error),
                            "recorded_at": now_iso(),
                        },
                        exclusive=True,
                    )
            except OSError:
                pass
        print(f"FERRUM RUNTIME VNEXT CUDA DETERMINISM COLLECTION REJECT: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
