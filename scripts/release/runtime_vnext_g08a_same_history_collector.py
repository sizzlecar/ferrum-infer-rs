#!/usr/bin/env python3
"""Collect resumable 20x64 G08A canonical-history numerical evidence."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import runtime_vnext_g08a_numerics as numerics
import runtime_vnext_g08a_same_history as same_history
import runtime_vnext_g08a_token_parity_collector as token_collector
import runtime_vnext_numerical_tolerances as tolerances


PASS_PREFIX = "FERRUM RUNTIME VNEXT G08A SAME HISTORY COLLECTOR PASS"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT G08A SAME HISTORY COLLECTOR FAIL"
SELFTEST_PASS = "FERRUM RUNTIME VNEXT G08A SAME HISTORY COLLECTOR SELFTEST PASS"
SCHEMA_VERSION = 1
THREAD_LIMIT = 4
HELPER_BUILD_TIMEOUT_SECONDS = 600
FERRUM_TIMEOUT_SECONDS = 300
LLAMA_TIMEOUT_SECONDS = 180
ORACLE_TIMEOUT_SECONDS = 240
COMPARISON_TIMEOUT_SECONDS = 180
VALIDATION_TIMEOUT_SECONDS = 180
NUMPY_REQUIREMENT = "numpy==2.4.6"
PYYAML_REQUIREMENT = "pyyaml==6.0.3"

REPO_ROOT = Path(__file__).resolve().parents[2]
BOUNDED_COMMAND = REPO_ROOT / "scripts/release/bounded_command.py"
HELPER_SOURCE = REPO_ROOT / "scripts/release/llama_teacher_logits_dump.cpp"
ORACLE_SOURCE = REPO_ROOT / "scripts/release/qwen35_gguf_teacher_logits_reference.py"
VALIDATOR_SOURCE = REPO_ROOT / "scripts/release/runtime_vnext_g08a_same_history.py"

ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "source_git_sha",
        "source_tree_sha",
        "source_dirty",
        "model_id",
        "model_revision",
        "model_file_sha256",
        "backend",
        "token_parity",
        "tolerance_catalog",
        "ferrum_binary",
        "llama_cpp_source",
        "llama_helper",
        "case_count",
        "decision_count_per_case",
        "validated_decision_count",
        "summary",
        "exception_count",
        "waiver_count",
        "cases",
    }
)
CASE_FIELDS = frozenset(
    {
        "prompt_id",
        "prompt_sha256",
        "prompt_token_ids_sha256",
        "teacher_token_ids_sha256",
        "cache_key",
        "manifest",
        "stages",
        "validation",
        "summary",
    }
)
SUMMARY_FIELDS = frozenset(
    {
        "robust_decision_count",
        "ambiguous_decision_count",
        "ferrum_oracle_exact_count",
        "ambiguous_top2_accepted_count",
        "llama_oracle_exact_count",
        "external_flip_count",
        "exception_count",
        "waiver_count",
    }
)
ARTIFACT_REF_FIELDS = frozenset({"path", "sha256"})
ENVIRONMENT_OVERRIDES = {
    "NO_COLOR": "1",
    "RAYON_NUM_THREADS": str(THREAD_LIMIT),
    "TOKIO_WORKER_THREADS": "2",
    "OMP_NUM_THREADS": str(THREAD_LIMIT),
    "OPENBLAS_NUM_THREADS": str(THREAD_LIMIT),
    "MKL_NUM_THREADS": str(THREAD_LIMIT),
    "NUMEXPR_NUM_THREADS": str(THREAD_LIMIT),
    "VECLIB_MAXIMUM_THREADS": str(THREAD_LIMIT),
}


class CollectorError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectorError(message)


def exact_object(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == fields, f"{label} fields differ: {sorted(set(value) ^ fields)}")
    return value


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CollectorError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def artifact_ref(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    require(resolved.is_file(), f"artifact file is missing: {resolved}")
    return {"path": str(resolved), "sha256": token_collector.sha256_file(resolved)}


def validate_artifact_ref(value: Any, label: str) -> Path:
    record = exact_object(value, ARTIFACT_REF_FIELDS, label)
    path = Path(record["path"]).expanduser().resolve()
    require(path.is_file(), f"{label} is missing: {path}")
    require(token_collector.sha256_file(path) == record["sha256"], f"{label} SHA256 differs")
    return path


def clean_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("FERRUM_")
    }
    environment.update(ENVIRONMENT_OVERRIDES)
    return environment


def stage_record(root: Path) -> dict[str, Any]:
    return {
        "command": artifact_ref(root / "command.json"),
        "receipt": artifact_ref(root / "receipt.json"),
        "stdout": artifact_ref(root / "stdout.log"),
        "stderr": artifact_ref(root / "stderr.log"),
    }


def validate_stage_record(value: Any, label: str) -> None:
    record = exact_object(value, frozenset({"command", "receipt", "stdout", "stderr"}), label)
    command_path = validate_artifact_ref(record["command"], f"{label}.command")
    receipt_path = validate_artifact_ref(record["receipt"], f"{label}.receipt")
    validate_artifact_ref(record["stdout"], f"{label}.stdout")
    validate_artifact_ref(record["stderr"], f"{label}.stderr")
    receipt = read_object(receipt_path, f"{label} bounded receipt")
    command = read_object(command_path, f"{label} command")
    limits = receipt.get("limits")
    require(
        receipt.get("schema") == "ferrum.bounded-command-receipt.v1"
        and receipt.get("status") == "pass"
        and receipt.get("rc") == 0,
        f"{label} bounded receipt is not PASS",
    )
    require(
        receipt.get("command") == command.get("argv")
        and receipt.get("cwd") == command.get("cwd")
        and isinstance(limits, dict)
        and limits.get("wall_timeout_seconds") == float(command.get("hard_deadline_seconds"))
        and limits.get("max_processes") == 16
        and limits.get("max_group_threads") == 32
        and limits.get("max_per_process_threads") == 8,
        f"{label} command or resource envelope differs",
    )


def stage_contract(
    *,
    argv: list[str],
    timeout_seconds: int,
    progress_signal: str,
    input_key: str,
) -> dict[str, Any]:
    return {
        "argv": argv,
        "cwd": str(REPO_ROOT),
        "environment_overrides": ENVIRONMENT_OVERRIDES,
        "removed_environment_prefixes": ["FERRUM_"],
        "expected_duration_seconds": timeout_seconds // 2,
        "hard_deadline_seconds": timeout_seconds,
        "progress_signal": progress_signal,
        "input_key": input_key,
    }


def run_stage(
    *,
    root: Path,
    argv: list[str],
    timeout_seconds: int,
    progress_signal: str,
    input_key: str,
) -> dict[str, Any]:
    require(not root.exists(), f"stage output already exists: {root}")
    root.mkdir(parents=True)
    token_collector.atomic_write_json(
        root / "command.json",
        stage_contract(
            argv=argv,
            timeout_seconds=timeout_seconds,
            progress_signal=progress_signal,
            input_key=input_key,
        ),
    )
    bounded_argv = [
        sys.executable,
        str(BOUNDED_COMMAND),
        "--receipt",
        str(root / "receipt.json"),
        "--stdout-log",
        str(root / "stdout.log"),
        "--stderr-log",
        str(root / "stderr.log"),
        "--cwd",
        str(REPO_ROOT),
        "--wall-timeout-seconds",
        str(timeout_seconds),
        "--max-processes",
        "16",
        "--max-group-threads",
        "32",
        "--max-per-process-threads",
        "8",
        "--",
        *argv,
    ]
    print(f"stage {root.name}: start; deadline={timeout_seconds}s", flush=True)
    runner_path = root / "runner.log"
    started = time.monotonic()
    next_progress = started + 5
    with runner_path.open("wb") as runner:
        process = subprocess.Popen(
            bounded_argv,
            cwd=REPO_ROOT,
            env=clean_environment(),
            stdin=subprocess.DEVNULL,
            stdout=runner,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            time.sleep(0.5)
            now = time.monotonic()
            if now >= next_progress:
                stdout_bytes = (root / "stdout.log").stat().st_size if (root / "stdout.log").is_file() else 0
                stderr_bytes = (root / "stderr.log").stat().st_size if (root / "stderr.log").is_file() else 0
                print(
                    f"stage {root.name}: running elapsed={now - started:.0f}s "
                    f"stdout={stdout_bytes}B stderr={stderr_bytes}B",
                    flush=True,
                )
                next_progress = now + 5
        returncode = process.wait()
    require(
        returncode == 0,
        f"stage {root.name} failed with exit code {returncode}; see {root}",
    )
    record = stage_record(root)
    validate_stage_record(record, f"stage {root.name}")
    print(f"stage {root.name}: PASS", flush=True)
    return record


def run_or_reuse_stage(
    *,
    root: Path,
    argv: list[str],
    timeout_seconds: int,
    progress_signal: str,
    input_key: str,
    output_paths: list[Path],
    validate_output: Callable[[], None],
) -> dict[str, Any]:
    expected_contract = stage_contract(
        argv=argv,
        timeout_seconds=timeout_seconds,
        progress_signal=progress_signal,
        input_key=input_key,
    )
    if root.exists():
        try:
            require(read_object(root / "command.json", "cached stage command") == expected_contract, "cached stage command differs")
            record = stage_record(root)
            validate_stage_record(record, f"cached stage {root.name}")
            validate_output()
            print(f"stage {root.name}: cached PASS", flush=True)
            return record
        except (CollectorError, same_history.GateError, OSError) as error:
            print(f"stage {root.name}: cache rejected: {error}", flush=True)
            shutil.rmtree(root)
    for path in output_paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    record = run_stage(
        root=root,
        argv=argv,
        timeout_seconds=timeout_seconds,
        progress_signal=progress_signal,
        input_key=input_key,
    )
    validate_output()
    return record


def require_stdout(path: Path, expected_line: str, label: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    require(expected_line in lines, f"{label} exact PASS line is missing")


def source_file_identity(path: Path) -> dict[str, str]:
    require(path.is_file(), f"source file is missing: {path}")
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": token_collector.sha256_file(path)}


def library_identities(llama_cpp_root: Path) -> list[dict[str, Any]]:
    build_bin = llama_cpp_root / "build/bin"
    candidates = sorted(build_bin.glob("libggml*.dylib")) + sorted(build_bin.glob("libllama*.dylib"))
    resolved = sorted({candidate.resolve() for candidate in candidates})
    require(resolved, "llama.cpp build has no resolved dylib artifacts")
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": token_collector.sha256_file(path)}
        for path in resolved
    ]


def validate_libraries(value: Any) -> None:
    require(isinstance(value, list) and value, "llama helper libraries are missing")
    paths: list[str] = []
    for index, raw in enumerate(value):
        record = exact_object(raw, frozenset({"path", "bytes", "sha256"}), f"llama library[{index}]")
        path = Path(record["path"]).resolve()
        require(
            path.is_file()
            and path.stat().st_size == record["bytes"]
            and token_collector.sha256_file(path) == record["sha256"],
            f"llama library[{index}] identity differs",
        )
        paths.append(str(path))
    require(paths == sorted(set(paths)), "llama helper libraries must be unique and sorted")


def helper_key(llama_cpp: dict[str, Any]) -> str:
    return same_history.canonical_sha256(
        {
            "source_git_sha": llama_cpp["git_sha"],
            "helper_source": source_file_identity(HELPER_SOURCE),
            "thread_limit": THREAD_LIMIT,
            "compiler_flags": ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror"],
        }
    )


def validate_helper(value: Any, expected_key: str) -> Path:
    record = exact_object(
        value,
        frozenset({"cache_key", "source", "binary", "libraries", "cmake_build", "compile"}),
        "llama helper",
    )
    require(record["cache_key"] == expected_key, "llama helper cache key differs")
    require(record["source"] == source_file_identity(HELPER_SOURCE), "llama helper source identity differs")
    binary = validate_artifact_ref(record["binary"], "llama helper binary")
    require(os.access(binary, os.X_OK), "llama helper binary is not executable")
    validate_libraries(record["libraries"])
    validate_stage_record(record["cmake_build"], "llama cmake build")
    validate_stage_record(record["compile"], "llama helper compile")
    return binary


def build_helper(out: Path, llama_cpp: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    require(sys.platform == "darwin", "G08A same-history Metal collection requires macOS")
    root = out / "tools/llama-helper"
    key = helper_key(llama_cpp)
    result_path = root / "result.json"
    if result_path.is_file():
        try:
            record = read_object(result_path, "cached llama helper")
            return validate_helper(record, key), record
        except CollectorError as error:
            print(f"llama helper cache rejected: {error}", flush=True)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    source_root = Path(llama_cpp["path"])
    cmake_stage = run_stage(
        root=root / "cmake-build",
        argv=["cmake", "--build", str(source_root / "build"), "--target", "llama", "--parallel", str(THREAD_LIMIT)],
        timeout_seconds=HELPER_BUILD_TIMEOUT_SECONDS,
        progress_signal="bounded receipt samples plus cmake stdout bytes",
        input_key=f"{key}:cmake-build",
    )
    libraries = library_identities(source_root)
    binary = root / "llama_teacher_logits_dump"
    compile_stage = run_stage(
        root=root / "compile",
        argv=[
            "c++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            f"-I{source_root / 'include'}",
            f"-I{source_root / 'ggml/include'}",
            str(HELPER_SOURCE),
            f"-L{source_root / 'build/bin'}",
            f"-Wl,-rpath,{source_root / 'build/bin'}",
            "-lllama",
            "-lggml",
            "-lggml-base",
            "-o",
            str(binary),
        ],
        timeout_seconds=HELPER_BUILD_TIMEOUT_SECONDS,
        progress_signal="bounded receipt samples plus compiler stdout bytes",
        input_key=same_history.canonical_sha256({"helper_key": key, "libraries": libraries}),
    )
    require(binary.is_file() and os.access(binary, os.X_OK), "llama helper compilation produced no executable")
    record = {
        "cache_key": key,
        "source": source_file_identity(HELPER_SOURCE),
        "binary": artifact_ref(binary),
        "libraries": libraries,
        "cmake_build": cmake_stage,
        "compile": compile_stage,
    }
    token_collector.atomic_write_json(result_path, record)
    validate_helper(record, key)
    return binary, record


def ferrum_argv(binary: Path, prompt: str, root: Path, teacher_path: Path) -> list[str]:
    argv = token_collector.ferrum_argv(binary, prompt, root / "request-dump")
    return [
        *argv,
        "--effective-config-json",
        str(root / "effective-config.json"),
        "--decision-trace-jsonl",
        str(root / "decision.jsonl"),
        "--vnext-checkpoint-dir",
        str(root / "capture"),
        "--vnext-checkpoint-product-output",
        "--vnext-checkpoint-teacher-token-file",
        str(teacher_path),
        "--vnext-checkpoint-prefill-waves",
        "1",
        "--vnext-checkpoint-decode-waves",
        str(same_history.DECISION_COUNT - 1),
    ]


def case_cache_key(context: dict[str, Any], parity_case: dict[str, Any]) -> str:
    return same_history.canonical_sha256(
        {
            "source_git_sha": context["source_git_sha"],
            "source_tree_sha": context["source_tree_sha"],
            "ferrum_binary_sha256": context["ferrum_binary"]["sha256"],
            "llama_source_git_sha": context["llama_cpp_source"]["git_sha"],
            "llama_helper_sha256": context["llama_helper"]["binary"]["sha256"],
            "model_file_sha256": context["model_file_sha256"],
            "parity_case": parity_case,
            "oracle_source": source_file_identity(ORACLE_SOURCE),
            "validator_source": source_file_identity(VALIDATOR_SOURCE),
            "tolerance_catalog_sha256": context["tolerance_catalog_sha256"],
            "tolerance_row_fingerprint": context["tolerance_row"]["row_fingerprint"],
            "thread_limit": THREAD_LIMIT,
        }
    )


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, int]:
    result = {field: 0 for field in SUMMARY_FIELDS}
    for index, raw in enumerate(summaries):
        summary = exact_object(raw, SUMMARY_FIELDS, f"case summary[{index}]")
        for field in SUMMARY_FIELDS:
            value = summary[field]
            require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"case summary[{index}].{field} is invalid")
            result[field] += value
    return result


def create_case_manifest(
    *,
    context: dict[str, Any],
    parity_case: dict[str, Any],
    ferrum_capture: Path,
    llama_capture: Path,
    oracle_dir: Path,
    comparison: Path,
) -> dict[str, Any]:
    return {
        "schema_version": same_history.SCHEMA_VERSION,
        "status": "pass",
        "source_git_sha": context["source_git_sha"],
        "source_tree_sha": context["source_tree_sha"],
        "source_dirty": False,
        "model_id": same_history.MODEL_ID,
        "model_revision": context["model_revision"],
        "model_file_sha256": context["model_file_sha256"],
        "backend": "metal",
        "tolerance_id": same_history.TOLERANCE_ID,
        "tolerance_catalog_sha256": context["tolerance_catalog_sha256"],
        "tolerance_row_fingerprint": context["tolerance_row"]["row_fingerprint"],
        "oracle_ambiguity_margin": same_history.ORACLE_AMBIGUITY_MARGIN,
        "prompt_id": parity_case["prompt_id"],
        "prompt_sha256": parity_case["prompt_sha256"],
        "prompt_token_ids_sha256": parity_case["prompt_token_ids_sha256"],
        "teacher_token_ids_sha256": parity_case["reference_generated_token_ids_sha256"],
        "teacher_token_count": same_history.DECISION_COUNT,
        "ferrum_capture_dir": str(ferrum_capture.resolve()),
        "llama_capture_dir": str(llama_capture.resolve()),
        "oracle_dir": str(oracle_dir.resolve()),
        "comparison": str(comparison.resolve()),
        "exception_count": 0,
        "waiver_count": 0,
    }


def collect_case(
    *,
    context: dict[str, Any],
    parity_case: dict[str, Any],
    prompt: dict[str, Any],
    case_root: Path,
) -> dict[str, Any]:
    teacher_tokens = parity_case["reference_generated_token_ids"]
    prompt_tokens = parity_case["reference_prompt_token_ids"]
    require(len(teacher_tokens) == same_history.DECISION_COUNT, "teacher token denominator differs")
    teacher_path = case_root / "teacher-token-ids.json"
    token_collector.atomic_write_json(
        teacher_path,
        {"schema_version": 1, "encoding": "u32-le", "token_ids": teacher_tokens},
    )
    cache_key = case_cache_key(context, parity_case)
    content = prompt["messages"][0]["content"]
    ferrum_root = case_root / "ferrum"
    ferrum_capture = ferrum_root / "capture"

    def validate_ferrum_output() -> None:
        _paths, captured_prompt, captured_teacher, _summary = same_history.ferrum_paths_and_history(ferrum_capture)
        require(captured_prompt == prompt_tokens, "Ferrum captured prompt tokens differ from token parity")
        require(captured_teacher == teacher_tokens, "Ferrum captured teacher tokens differ from token parity")

    ferrum_command = ferrum_argv(Path(context["ferrum_binary"]["path"]), content, ferrum_root, teacher_path)
    ferrum_stage = run_or_reuse_stage(
        root=case_root / "stage-ferrum",
        argv=ferrum_command,
        timeout_seconds=FERRUM_TIMEOUT_SECONDS,
        progress_signal="capture plan.json plus wave files and bounded receipt samples",
        input_key=same_history.canonical_sha256(
            {
                "case_key": cache_key,
                "command": ferrum_command,
                "teacher_file_sha256": token_collector.sha256_file(teacher_path),
            }
        ),
        output_paths=[ferrum_root],
        validate_output=validate_ferrum_output,
    )
    ferrum_paths, _captured_prompt, _captured_teacher, _ferrum_summary = same_history.ferrum_paths_and_history(ferrum_capture)

    llama_input = case_root / "llama-input-token-ids.txt"
    token_collector.atomic_write(
        llama_input,
        (" ".join(str(token) for token in prompt_tokens + teacher_tokens[:-1]) + "\n").encode("ascii"),
    )
    llama_capture = case_root / "llama-capture"
    llama_command = [
        context["llama_helper"]["binary"]["path"],
        str(context["model_path"]),
        str(llama_input),
        str(len(prompt_tokens)),
        str(llama_capture),
    ]

    def validate_llama_output() -> None:
        require_stdout(
            case_root / "stage-llama/stdout.log",
            f"LLAMA TEACHER LOGITS DUMP PASS: {llama_capture}",
            "llama helper",
        )
        same_history.validate_llama_manifest(
            llama_capture,
            prompt_tokens=prompt_tokens,
            teacher_tokens=teacher_tokens,
        )

    llama_stage = run_or_reuse_stage(
        root=case_root / "stage-llama",
        argv=llama_command,
        timeout_seconds=LLAMA_TIMEOUT_SECONDS,
        progress_signal="decision-*.f32 file count and bounded receipt samples",
        input_key=same_history.canonical_sha256(
            {
                "case_key": cache_key,
                "command": llama_command,
                "input_sha256": token_collector.sha256_file(llama_input),
                "helper_sha256": context["llama_helper"]["binary"]["sha256"],
            }
        ),
        output_paths=[llama_capture],
        validate_output=validate_llama_output,
    )
    llama_paths, _llama_manifest = same_history.validate_llama_manifest(
        llama_capture,
        prompt_tokens=prompt_tokens,
        teacher_tokens=teacher_tokens,
    )

    oracle_dir = case_root / "oracle"
    oracle_command = [
        "uv",
        "run",
        "--with",
        NUMPY_REQUIREMENT,
        "--with",
        PYYAML_REQUIREMENT,
        "python3",
        str(ORACLE_SOURCE),
        "--model",
        str(context["model_path"]),
        "--prompt-token-ids",
        str(ferrum_capture / "teacher-prompt.json"),
        "--teacher-token-ids",
        str(teacher_path),
        "--llama-cpp-root",
        context["llama_cpp_source"]["path"],
        "--out",
        str(oracle_dir),
    ]

    def validate_oracle_output() -> None:
        require_stdout(
            case_root / "stage-oracle/stdout.log",
            f"QWEN35 GGUF TEACHER LOGITS REFERENCE PASS: {oracle_dir}",
            "CPU oracle",
        )
        same_history.validate_oracle_report(
            oracle_dir,
            source_git_sha=context["source_git_sha"],
            prompt_token_sha256=parity_case["prompt_token_ids_sha256"],
            teacher_token_sha256=parity_case["reference_generated_token_ids_sha256"],
        )

    oracle_stage = run_or_reuse_stage(
        root=case_root / "stage-oracle",
        argv=oracle_command,
        timeout_seconds=ORACLE_TIMEOUT_SECONDS,
        progress_signal="oracle report/raw logits bytes and bounded receipt samples",
        input_key=same_history.canonical_sha256(
            {
                "case_key": cache_key,
                "command": oracle_command,
                "prompt_file_sha256": token_collector.sha256_file(ferrum_capture / "teacher-prompt.json"),
                "teacher_file_sha256": token_collector.sha256_file(teacher_path),
                "oracle_source": source_file_identity(ORACLE_SOURCE),
            }
        ),
        output_paths=[oracle_dir],
        validate_output=validate_oracle_output,
    )
    oracle_path, oracle_decisions, _oracle_report = same_history.validate_oracle_report(
        oracle_dir,
        source_git_sha=context["source_git_sha"],
        prompt_token_sha256=parity_case["prompt_token_ids_sha256"],
        teacher_token_sha256=parity_case["reference_generated_token_ids_sha256"],
    )

    comparison = case_root / "comparison.json"
    comparison_command = [
        "uv",
        "run",
        "--with",
        NUMPY_REQUIREMENT,
        "python3",
        str(VALIDATOR_SOURCE),
        "--build-comparison",
        "--ferrum-capture",
        str(ferrum_capture),
        "--llama-capture",
        str(llama_capture),
        "--oracle-dir",
        str(oracle_dir),
        "--out",
        str(comparison),
    ]
    bounds = same_history.validate_tolerance_row(context["tolerance_row"])

    def validate_comparison_output() -> None:
        require(comparison.is_file(), "same-history comparison output is missing")
        same_history.validate_comparison(
            comparison,
            ferrum_paths=ferrum_paths,
            llama_paths=llama_paths,
            oracle_path=oracle_path,
            oracle_decisions=oracle_decisions,
            bounds=bounds,
        )

    comparison_stage = run_or_reuse_stage(
        root=case_root / "stage-comparison",
        argv=comparison_command,
        timeout_seconds=COMPARISON_TIMEOUT_SECONDS,
        progress_signal="comparison.json bytes and bounded receipt samples",
        input_key=same_history.canonical_sha256(
            {
                "case_key": cache_key,
                "command": comparison_command,
                "ferrum_raw_sha256": same_history.ordered_raw_sha256(ferrum_paths),
                "llama_raw_sha256": same_history.ordered_raw_sha256(llama_paths),
                "oracle_raw_sha256": token_collector.sha256_file(oracle_path),
                "tolerance_row_fingerprint": context["tolerance_row"]["row_fingerprint"],
            }
        ),
        output_paths=[comparison],
        validate_output=validate_comparison_output,
    )

    manifest_path = case_root / "manifest.json"
    manifest = create_case_manifest(
        context=context,
        parity_case=parity_case,
        ferrum_capture=ferrum_capture,
        llama_capture=llama_capture,
        oracle_dir=oracle_dir,
        comparison=comparison,
    )
    token_collector.atomic_write_json(manifest_path, manifest)
    validation_summary: dict[str, Any] = {}

    def validate_final_output() -> None:
        require_stdout(
            case_root / "stage-validation/stdout.log",
            f"{same_history.PASS_PREFIX}: {manifest_path.resolve()}",
            "same-history validator",
        )
        validation_summary.clear()
        validation_summary.update(
            same_history.validate_manifest(
                manifest_path,
                context["tolerance_row"],
                context["tolerance_catalog_sha256"],
            )
        )

    validation_command = ["python3", str(VALIDATOR_SOURCE), "--manifest", str(manifest_path)]
    validation_stage = run_or_reuse_stage(
        root=case_root / "stage-validation",
        argv=validation_command,
        timeout_seconds=VALIDATION_TIMEOUT_SECONDS,
        progress_signal="exact same-history PASS line and bounded receipt samples",
        input_key=same_history.canonical_sha256(
            {
                "case_key": cache_key,
                "command": validation_command,
                "manifest_sha256": token_collector.sha256_file(manifest_path),
                "comparison_sha256": token_collector.sha256_file(comparison),
                "catalog_sha256": context["tolerance_catalog_sha256"],
            }
        ),
        output_paths=[],
        validate_output=validate_final_output,
    )
    summary = validation_summary
    result = {
        "prompt_id": parity_case["prompt_id"],
        "prompt_sha256": parity_case["prompt_sha256"],
        "prompt_token_ids_sha256": parity_case["prompt_token_ids_sha256"],
        "teacher_token_ids_sha256": parity_case["reference_generated_token_ids_sha256"],
        "cache_key": cache_key,
        "manifest": artifact_ref(manifest_path),
        "stages": {
            "ferrum": ferrum_stage,
            "llama": llama_stage,
            "oracle": oracle_stage,
            "comparison": comparison_stage,
            "validation": validation_stage,
        },
        "validation": validation_stage,
        "summary": {field: summary[field] for field in SUMMARY_FIELDS},
    }
    token_collector.atomic_write_json(case_root / "case.result.json", result)
    return result


def validate_case(
    value: Any,
    *,
    context: dict[str, Any],
    parity_case: dict[str, Any],
) -> dict[str, Any]:
    record = exact_object(value, CASE_FIELDS, "same-history case")
    require(record["cache_key"] == case_cache_key(context, parity_case), "same-history case cache key differs")
    for field in ("prompt_id", "prompt_sha256", "prompt_token_ids_sha256"):
        require(record[field] == parity_case[field], f"same-history case {field} differs")
    require(
        record["teacher_token_ids_sha256"] == parity_case["reference_generated_token_ids_sha256"],
        "same-history case teacher token SHA differs",
    )
    manifest_path = validate_artifact_ref(record["manifest"], "same-history case manifest")
    stages = exact_object(
        record["stages"],
        frozenset({"ferrum", "llama", "oracle", "comparison", "validation"}),
        "same-history case stages",
    )
    for stage_name, stage in stages.items():
        validate_stage_record(stage, f"same-history case stage {stage_name}")
    require(record["validation"] == stages["validation"], "same-history validation stage differs")
    require_stdout(
        Path(record["validation"]["stdout"]["path"]),
        f"{same_history.PASS_PREFIX}: {manifest_path}",
        "same-history cached validator",
    )
    summary = same_history.validate_manifest(
        manifest_path,
        context["tolerance_row"],
        context["tolerance_catalog_sha256"],
    )
    require(record["summary"] == {field: summary[field] for field in SUMMARY_FIELDS}, "same-history case summary differs")
    return record


def collection_context(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    source_git_sha, source_tree_sha, _catalog_blob, _catalog_summary = numerics.current_source_identity(require_clean=True)
    parity = read_object(args.token_parity.expanduser().resolve(), "token parity")
    token_summary = numerics.validate_token_parity(args.token_parity, source_git_sha, source_tree_sha)
    ferrum_binary = token_collector.binary_identity(args.ferrum_binary, "Ferrum binary")
    require(ferrum_binary == parity["ferrum_binary"], "same-history Ferrum binary differs from token parity")
    llama_cpp_source = token_collector.clean_git_identity(args.llama_cpp_source, "llama.cpp")
    require(llama_cpp_source["git_sha"] == parity["reference_source_git_sha"], "same-history llama.cpp source differs from token parity")
    identity = numerics.model_lock_identity()
    model_path = token_collector.locked_model_path(args.model, identity["model_revision"], identity["model_file_sha256"])
    tolerance_row, tolerance_catalog_sha256 = same_history.load_tolerance_row(tolerances.DEFAULT_CATALOG)
    context: dict[str, Any] = {
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "ferrum_binary": ferrum_binary,
        "llama_cpp_source": llama_cpp_source,
        "model_path": model_path,
        **identity,
        "tolerance_row": tolerance_row,
        "tolerance_catalog_sha256": tolerance_catalog_sha256,
        "token_parity_path": args.token_parity.expanduser().resolve(),
        "token_parity_summary": token_summary,
    }
    return context, parity


def validate_collection(path: Path, *, require_clean: bool = True) -> dict[str, Any]:
    document = exact_object(read_object(path, "same-history collection"), ROOT_FIELDS, "same-history collection")
    require(document["schema_version"] == SCHEMA_VERSION and document["status"] == "pass", "same-history collection is not PASS")
    source_git_sha, source_tree_sha, _blob, _summary = numerics.current_source_identity(require_clean=require_clean)
    require(
        document["source_git_sha"] == source_git_sha
        and document["source_tree_sha"] == source_tree_sha
        and document["source_dirty"] is False,
        "same-history collection source is stale or dirty",
    )
    identity = numerics.model_lock_identity()
    require(
        document["model_id"] == same_history.MODEL_ID
        and document["model_revision"] == identity["model_revision"]
        and document["model_file_sha256"] == identity["model_file_sha256"]
        and document["backend"] == "metal",
        "same-history collection model/backend differs",
    )
    token_parity_path = validate_artifact_ref(document["token_parity"], "same-history token parity")
    parity = read_object(token_parity_path, "same-history token parity")
    numerics.validate_token_parity(token_parity_path, source_git_sha, source_tree_sha)
    ferrum_binary = exact_object(document["ferrum_binary"], ARTIFACT_REF_FIELDS, "same-history Ferrum binary")
    validate_artifact_ref(ferrum_binary, "same-history Ferrum binary")
    require(ferrum_binary == parity["ferrum_binary"], "same-history Ferrum binary differs from token parity")
    llama_cpp = exact_object(document["llama_cpp_source"], frozenset({"path", "git_sha", "dirty"}), "same-history llama.cpp source")
    current_llama = token_collector.clean_git_identity(Path(llama_cpp["path"]), "llama.cpp")
    require(current_llama == llama_cpp and llama_cpp["git_sha"] == parity["reference_source_git_sha"], "same-history llama.cpp identity differs")
    tolerance_row, catalog_sha = same_history.load_tolerance_row(tolerances.DEFAULT_CATALOG)
    tolerance_record = exact_object(document["tolerance_catalog"], frozenset({"path", "sha256", "row_fingerprint"}), "same-history tolerance catalog")
    require(
        Path(tolerance_record["path"]).resolve() == tolerances.DEFAULT_CATALOG.resolve()
        and tolerance_record["sha256"] == catalog_sha
        and tolerance_record["row_fingerprint"] == tolerance_row["row_fingerprint"],
        "same-history tolerance catalog identity differs",
    )
    helper_record = document["llama_helper"]
    helper_binary = validate_helper(helper_record, helper_key(llama_cpp))
    context = {
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "ferrum_binary": ferrum_binary,
        "llama_cpp_source": llama_cpp,
        "llama_helper": helper_record,
        "model_path": Path("MODEL_PATH_NOT_REQUIRED_FOR_VALIDATION"),
        **identity,
        "tolerance_row": tolerance_row,
        "tolerance_catalog_sha256": catalog_sha,
    }
    require(helper_binary == Path(helper_record["binary"]["path"]).resolve(), "same-history helper path differs")
    cases = document["cases"]
    require(isinstance(cases, list) and len(cases) == numerics.PROMPT_COUNT, "same-history collection must contain 20 cases")
    parity_cases = parity["cases"]
    require([case.get("prompt_id") for case in cases if isinstance(case, dict)] == [case["prompt_id"] for case in parity_cases], "same-history case order differs")
    validated = [validate_case(case, context=context, parity_case=parity_case) for case, parity_case in zip(cases, parity_cases, strict=True)]
    aggregate = aggregate_summaries([case["summary"] for case in validated])
    require(document["summary"] == aggregate, "same-history aggregate summary differs")
    require(
        document["case_count"] == numerics.PROMPT_COUNT
        and document["decision_count_per_case"] == same_history.DECISION_COUNT
        and document["validated_decision_count"] == numerics.PROMPT_COUNT * same_history.DECISION_COUNT,
        "same-history collection denominator differs",
    )
    require(
        aggregate["robust_decision_count"] + aggregate["ambiguous_decision_count"] == document["validated_decision_count"]
        and aggregate["exception_count"] == aggregate["waiver_count"] == 0
        and document["exception_count"] == document["waiver_count"] == 0,
        "same-history collection contains incomplete decisions or waiver/exception",
    )
    return {
        "case_count": document["case_count"],
        "validated_decision_count": document["validated_decision_count"],
        **aggregate,
    }


def collect(args: argparse.Namespace) -> Path:
    out = args.out.expanduser().resolve()
    require(not out.is_relative_to(REPO_ROOT), "same-history artifacts must be outside the source tree")
    context, parity = collection_context(args)
    out.mkdir(parents=True, exist_ok=True)
    helper_binary, helper_record = build_helper(out, context["llama_cpp_source"])
    context["llama_helper"] = helper_record
    require(helper_binary == Path(helper_record["binary"]["path"]), "llama helper path differs")
    corpus = token_collector.validate_corpus()
    prompts = {prompt["id"]: prompt for prompt in corpus}
    results: list[dict[str, Any]] = []
    for ordinal, parity_case in enumerate(parity["cases"], start=1):
        prompt_id = parity_case["prompt_id"]
        case_root = out / "cases" / prompt_id
        result_path = case_root / "case.result.json"
        if result_path.is_file():
            try:
                result = validate_case(read_object(result_path, "cached case"), context=context, parity_case=parity_case)
                print(f"same-history {prompt_id}: cached PASS ({ordinal}/{numerics.PROMPT_COUNT})", flush=True)
                results.append(result)
                continue
            except (CollectorError, same_history.GateError) as error:
                print(f"same-history {prompt_id}: cache rejected: {error}", flush=True)
        case_root.mkdir(parents=True, exist_ok=True)
        print(f"same-history {prompt_id}: collect ({ordinal}/{numerics.PROMPT_COUNT})", flush=True)
        result = collect_case(context=context, parity_case=parity_case, prompt=prompts[prompt_id], case_root=case_root)
        results.append(result)
        print(f"same-history {prompt_id}: PASS 64/64", flush=True)
    aggregate = aggregate_summaries([case["summary"] for case in results])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "source_git_sha": context["source_git_sha"],
        "source_tree_sha": context["source_tree_sha"],
        "source_dirty": False,
        "model_id": same_history.MODEL_ID,
        "model_revision": context["model_revision"],
        "model_file_sha256": context["model_file_sha256"],
        "backend": "metal",
        "token_parity": artifact_ref(context["token_parity_path"]),
        "tolerance_catalog": {
            "path": str(tolerances.DEFAULT_CATALOG.resolve()),
            "sha256": context["tolerance_catalog_sha256"],
            "row_fingerprint": context["tolerance_row"]["row_fingerprint"],
        },
        "ferrum_binary": context["ferrum_binary"],
        "llama_cpp_source": context["llama_cpp_source"],
        "llama_helper": helper_record,
        "case_count": numerics.PROMPT_COUNT,
        "decision_count_per_case": same_history.DECISION_COUNT,
        "validated_decision_count": numerics.PROMPT_COUNT * same_history.DECISION_COUNT,
        "summary": aggregate,
        "exception_count": 0,
        "waiver_count": 0,
        "cases": results,
    }
    manifest_path = out / "same-history.json"
    token_collector.atomic_write_json(manifest_path, manifest)
    validate_collection(manifest_path)
    return manifest_path


def self_test() -> None:
    base = {field: 0 for field in SUMMARY_FIELDS}
    first = copy.deepcopy(base)
    first.update({"robust_decision_count": 63, "ambiguous_decision_count": 1, "ferrum_oracle_exact_count": 64, "ambiguous_top2_accepted_count": 1, "llama_oracle_exact_count": 63, "external_flip_count": 1})
    second = copy.deepcopy(base)
    second.update({"robust_decision_count": 64, "ferrum_oracle_exact_count": 64, "llama_oracle_exact_count": 64})
    aggregate = aggregate_summaries([first, second])
    require(aggregate["robust_decision_count"] == 127 and aggregate["ambiguous_decision_count"] == 1, "summary aggregation differs")
    teacher = list(range(same_history.DECISION_COUNT))
    prompt = [100, 101]
    require(len(prompt + teacher[:-1]) == 65, "llama canonical-history off-by-one differs")
    argv = ferrum_argv(Path("/tmp/ferrum"), "hello", Path("/tmp/case/ferrum"), Path("/tmp/teacher.json"))
    require(
        argv.count("--vnext-checkpoint-product-output") == 1
        and argv[argv.index("--vnext-checkpoint-prefill-waves") + 1] == "1"
        and argv[argv.index("--vnext-checkpoint-decode-waves") + 1] == "63",
        "Ferrum canonical-history command contract differs",
    )
    context = {
        "source_git_sha": "1" * 40,
        "source_tree_sha": "2" * 40,
        "ferrum_binary": {"sha256": "3" * 64},
        "llama_cpp_source": {"git_sha": "4" * 40},
        "llama_helper": {"binary": {"sha256": "5" * 64}},
        "model_file_sha256": "6" * 64,
        "tolerance_catalog_sha256": "7" * 64,
        "tolerance_row": {"row_fingerprint": "8" * 64},
    }
    parity_case = {
        "prompt_id": "fixture",
        "reference_generated_token_ids": teacher,
        "reference_generated_token_ids_sha256": numerics.token_sha256(teacher),
    }
    key = case_cache_key(context, parity_case)
    changed = copy.deepcopy(parity_case)
    changed["reference_generated_token_ids"][0] = 999
    require(key != case_cache_key(context, changed), "case cache key ignores teacher history")
    with tempfile.TemporaryDirectory(prefix="g08a-same-history-collector-selftest-") as temporary:
        root = Path(temporary)
        output = root / "output.txt"
        stage = root / "stage"
        command = [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(output)!r}).write_text('ok', encoding='ascii')",
        ]

        def validate_output() -> None:
            require(output.is_file() and output.read_text(encoding="ascii") == "ok", "stage cache output differs")

        first_stage = run_or_reuse_stage(
            root=stage,
            argv=command,
            timeout_seconds=10,
            progress_signal="output.txt bytes",
            input_key="a" * 64,
            output_paths=[output],
            validate_output=validate_output,
        )
        second_stage = run_or_reuse_stage(
            root=stage,
            argv=command,
            timeout_seconds=10,
            progress_signal="output.txt bytes",
            input_key="a" * 64,
            output_paths=[output],
            validate_output=validate_output,
        )
        require(first_stage == second_stage, "valid stage cache was not reused")
        output.write_text("tampered", encoding="ascii")
        run_or_reuse_stage(
            root=stage,
            argv=command,
            timeout_seconds=10,
            progress_signal="output.txt bytes",
            input_key="a" * 64,
            output_paths=[output],
            validate_output=validate_output,
        )
        validate_output()
    print(SELFTEST_PASS)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--self-test", action="store_true")
    result.add_argument("--token-parity", type=Path)
    result.add_argument("--ferrum-binary", type=Path)
    result.add_argument("--llama-cpp-source", type=Path)
    result.add_argument("--model", type=Path)
    result.add_argument("--out", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        missing = [
            flag
            for flag, value in (
                ("--token-parity", args.token_parity),
                ("--ferrum-binary", args.ferrum_binary),
                ("--llama-cpp-source", args.llama_cpp_source),
                ("--model", args.model),
                ("--out", args.out),
            )
            if value is None
        ]
        require(not missing, "missing required arguments: " + ", ".join(missing))
        manifest_path = collect(args)
        print(f"{PASS_PREFIX}: {manifest_path.parent}")
        return 0
    except (
        CollectorError,
        same_history.GateError,
        numerics.GateError,
        token_collector.CollectorError,
        tolerances.CatalogError,
        OSError,
        ValueError,
    ) as error:
        print(f"{FAIL_PREFIX}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
