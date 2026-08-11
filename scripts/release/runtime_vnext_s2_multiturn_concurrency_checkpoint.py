#!/usr/bin/env python3
"""Validate the focused S2 CUDA multi-turn and concurrency product sentinel."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from runtime_vnext_s2_stream_disconnect_checkpoint import (
    ValidationError,
    assert_clean_output,
    expected_tree_paths,
    file_sha256,
    iso_now,
    json_sha256,
    read_json,
    read_jsonl,
    read_text,
    require,
    resolve_member,
    validate_resource_balance,
    validate_self_hash,
    write_json,
)


PASS_PREFIX = "FERRUM RUNTIME VNEXT S2 MULTITURN CONCURRENCY PASS"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT S2 MULTITURN CONCURRENCY FAIL"
SELFTEST_PASS_LINE = "FERRUM RUNTIME VNEXT S2 MULTITURN CONCURRENCY SELFTEST PASS"
CHECKPOINT_ID = "runtime-vnext-s2-multiturn-concurrency-sentinel"
ROOT = Path(__file__).resolve().parent
RUNNER_PATH = ROOT / "run_scenarios.py"
HELPER_PATH = ROOT / "openai_concurrency_quality_regression.py"
SCENARIO_MANIFEST_PATH = ROOT / "scenarios/runtime_vnext_s2_multiturn_concurrency_cuda.json"
MODEL = "Qwen/Qwen3.5-4B"
SECRET = "banana"
HISTORICAL_CASE_ID = "H02.1"
RUN_NAME = "m1_s2_run_multiturn_recall"
SERVE_NAME = "m1_s2_serve_multiturn_recall"
CONCURRENCY_NAME = "m1_s2_serve_concurrency_quality"
RUN_PROMPTS = (
    "Remember the codeword banana. Reply exactly OK.",
    "Reply with only TWO.",
    "Reply with only THREE.",
    "Reply with only FOUR.",
    "What codeword did I ask you to remember? Answer with only the codeword.",
)
RUN_ANSWERS = ("OK", "TWO", "THREE", "FOUR", SECRET)
SCENARIOS = (
    (RUN_NAME, "run_multiturn"),
    (SERVE_NAME, "serve_multiturn_recall"),
    (CONCURRENCY_NAME, "serve_concurrency_quality"),
)
EXECUTION_PHASES = {
    "vnext.request_accepted",
    "vnext.plan_built",
    "vnext.frame_started",
    "vnext.operation_submitted",
    "vnext.frame_completed",
    "vnext.request_completed",
}
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QWEN35_CACHE_RE = re.compile(
    r"(?:^|/)models--Qwen--Qwen3\.5-4B/snapshots/[0-9a-f]{40}/?$"
)
EXPECTED_CONCURRENCY = (1, 4)
EXPECTED_SERVE_REQUESTS = 7
EXPECTED_RUN_REQUESTS = len(RUN_PROMPTS)
WORKER_LIMIT = 32
DOES_NOT_PROVE = [
    "full S2",
    "full G02",
    "full C01-C21",
    "Metal",
    "other models",
    "performance",
    "parallel GPU kernel execution",
    "release readiness",
]


def is_qwen35_model(value: Any) -> bool:
    return value == MODEL or (
        isinstance(value, str) and QWEN35_CACHE_RE.search(value) is not None
    )


def require_exact_visible_answer(value: Any, expected: str, label: str) -> str:
    require(isinstance(value, str), f"{label}: assistant content is not text")
    lowered = value.lower()
    require("<think" not in lowered and "</think>" not in lowered, f"{label}: thinking tag leaked")
    require(value.strip() == expected, f"{label}: expected exact answer {expected!r}")
    return value


def redacted_request_body(value: dict[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(value)
    messages = body.get("messages")
    require(isinstance(messages, list), "request sidecar messages missing")
    for index, message in enumerate(messages):
        require(isinstance(message, dict), f"request sidecar message {index} invalid")
        content = message.get("content")
        require(isinstance(content, str), f"request sidecar message {index} content invalid")
        message["content"] = "[redacted]"
        message["content_redacted"] = True
        message["content_chars"] = len(content)
    return body


def validate_usage(value: Any, label: str) -> dict[str, int]:
    require(isinstance(value, dict), f"{label}: usage missing")
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(key)
        require(isinstance(item, int) and not isinstance(item, bool) and item >= 0, f"{label}.{key}")
        result[key] = item
    require(result["completion_tokens"] > 0, f"{label}: no completion tokens")
    require(
        result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"],
        f"{label}: inconsistent usage",
    )
    return result


def validate_chat_response(path: Path, *, expected: str | None = None) -> dict[str, Any]:
    value = read_json(path)
    require(value.get("object") == "chat.completion", f"{path}: object mismatch")
    require(value.get("model") == MODEL, f"{path}: model mismatch")
    response_id = value.get("id")
    require(isinstance(response_id, str) and response_id, f"{path}: response id missing")
    choices = value.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, f"{path}: choices mismatch")
    choice = choices[0]
    require(isinstance(choice, dict) and choice.get("index") == 0, f"{path}: choice missing")
    require(choice.get("finish_reason") == "stop", f"{path}: finish reason mismatch")
    message = choice.get("message")
    require(isinstance(message, dict), f"{path}: message missing")
    require(message.get("role") == "assistant", f"{path}: message role mismatch")
    require(
        message.get("reasoning") in (None, "")
        and message.get("reasoning_content") in (None, ""),
        f"{path}: reasoning content leaked",
    )
    content = message.get("content")
    require(isinstance(content, str) and content.strip(), f"{path}: empty assistant content")
    assert_clean_output(str(path), content)
    if expected is not None:
        require_exact_visible_answer(content, expected, str(path))
    return {
        "id": response_id,
        "content": content,
        "usage": validate_usage(value.get("usage"), str(path)),
    }


def validate_tree(source: Path, recorded: Path) -> dict[str, Any]:
    for path in source.rglob("*"):
        require(not path.is_symlink(), f"artifact contains symlink: {path}")
    tree_path = source / "artifact_tree.json"
    tree = read_json(tree_path)
    validate_self_hash(tree, "artifact_tree.json")
    require(tree.get("schema_version") == 1, "artifact tree schema mismatch")
    require(tree.get("artifact_root") == str(recorded), "artifact tree root mismatch")
    rows = tree.get("files")
    require(isinstance(rows, list) and tree.get("file_count") == len(rows), "artifact tree count mismatch")
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and set(row) == {"path", "size", "sha256"}, f"artifact tree row {index}")
        relative = row.get("path")
        require(isinstance(relative, str) and relative and relative not in indexed, f"artifact tree path {index}")
        raw = Path(relative)
        require(not raw.is_absolute() and ".." not in raw.parts, f"artifact tree escape: {relative}")
        path = source / raw
        require(path.is_file() and not path.is_symlink(), f"artifact tree file missing: {relative}")
        require(row.get("size") == path.stat().st_size, f"artifact tree size mismatch: {relative}")
        require(row.get("sha256") == file_sha256(path), f"artifact tree SHA mismatch: {relative}")
        if path.suffix.lower() in {".json", ".jsonl", ".log", ".txt"}:
            assert_clean_output(relative, read_text(path))
        indexed[relative] = row
    actual = expected_tree_paths(source)
    require(set(indexed) == actual, "artifact tree coverage mismatch")
    required = {
        "summary.json",
        "execution_receipt.json",
        "observability_summary.json",
        "response_format_matrix_contract.json",
        "inputs/run_scenarios.py",
        "inputs/scenario_manifest.json",
        "inputs/openai_concurrency_quality_regression.py",
        f"{RUN_NAME}/result.json",
        f"{RUN_NAME}/command.json",
        f"{RUN_NAME}/input.txt",
        f"{RUN_NAME}/stdout.jsonl",
        f"{SERVE_NAME}/result.json",
        f"{SERVE_NAME}/turn1.request.json",
        f"{SERVE_NAME}/turn1.json",
        f"{SERVE_NAME}/turn2.request.json",
        f"{SERVE_NAME}/turn2.json",
        f"{CONCURRENCY_NAME}/result.json",
        f"{CONCURRENCY_NAME}/c1.quality.json",
        f"{CONCURRENCY_NAME}/c4.quality.json",
        f"{CONCURRENCY_NAME}/concurrency_quality_regression.json",
    }
    for concurrency in EXPECTED_CONCURRENCY:
        for index in range(concurrency):
            required.add(
                f"{CONCURRENCY_NAME}/c{concurrency}.quality.{index:03d}.request.json"
            )
            required.add(
                f"{CONCURRENCY_NAME}/c{concurrency}.quality.{index:03d}.response.txt"
            )
    require(required <= actual, f"artifact tree omits required evidence: {sorted(required - actual)}")
    return {"file_count": len(indexed), "sha256": file_sha256(tree_path)}


def validate_manifest(source: Path) -> dict[str, Any]:
    copied = source / "inputs/scenario_manifest.json"
    require(file_sha256(copied) == file_sha256(SCENARIO_MANIFEST_PATH), "artifact manifest differs from current source")
    manifest = read_json(copied)
    require(manifest == read_json(SCENARIO_MANIFEST_PATH), "artifact manifest semantic mismatch")
    require(
        manifest.get("goal_scope") == {"full_s2": False, "multiturn_concurrency_sentinel": True},
        "manifest scope mismatch",
    )
    require(manifest.get("backend") == "cuda" and manifest.get("model") == MODEL, "manifest model/backend mismatch")
    require(
        manifest.get("server")
        == {
            "args": ["--backend", "cuda", "--max-num-seqs", "1"],
            "mode": "start",
        },
        "manifest must align run and serve typed capacity",
    )
    require(
        manifest.get("observability")
        == {"enabled": True, "profile_detail": "basic", "profile_sample_rate": 1.0},
        "manifest observability mismatch",
    )
    scenarios = manifest.get("scenarios")
    require(isinstance(scenarios, list), "manifest scenarios missing")
    require(
        [(item.get("name"), item.get("type")) for item in scenarios if isinstance(item, dict)]
        == list(SCENARIOS),
        "manifest scenario identity/order mismatch",
    )
    require(scenarios[0].get("enable_thinking") is False, "run multi-turn must disable thinking")
    require(
        scenarios[0].get("historical_case_ids") == [HISTORICAL_CASE_ID],
        "run multi-turn historical case binding mismatch",
    )
    require(
        scenarios[0].get("use_default_max_tokens") is True
        and "max_tokens" not in scenarios[0],
        "H02.1 must exercise the user-visible default max-token budget",
    )
    require(
        scenarios[0].get("min_assistant_turns") == len(RUN_PROMPTS)
        and scenarios[0].get("prompts") == list(RUN_PROMPTS),
        "H02.1 five-turn prompt contract mismatch",
    )
    require(scenarios[1].get("enable_thinking") is False, "serve multi-turn must disable thinking")
    require(scenarios[2].get("enable_thinking") is False, "concurrency probe must disable thinking")
    require(scenarios[2].get("concurrency_cells") == [1, 4], "concurrency cells mismatch")
    return manifest


def validate_identity(source: Path, expected_git_sha: str | None) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    summary = read_json(source / "summary.json")
    require(summary.get("schema_version") == 1 and summary.get("status") == "pass", "summary status mismatch")
    require(summary.get("backend") == "cuda" and summary.get("model") == MODEL, "summary identity mismatch")
    git_sha = summary.get("git_sha")
    require(isinstance(git_sha, str) and GIT_SHA_RE.fullmatch(git_sha), "summary git SHA invalid")
    if expected_git_sha is not None:
        require(git_sha == expected_git_sha, "artifact git SHA differs from validation candidate")
    require(summary.get("dirty_status") == {"is_dirty": False, "status_short": []}, "source checkout was dirty")
    require(summary.get("scenario_count") == 3 and summary.get("manifest_scenario_count") == 3, "summary scenario count mismatch")
    require(summary.get("requested_scenarios") == [], "checkpoint must run full manifest")
    require(summary.get("selected_scenarios") == [name for name, _ in SCENARIOS], "selected scenario mismatch")
    require(summary.get("failed") == 0 and summary.get("skipped") == 0, "summary failure/skip count mismatch")
    recorded_value = summary.get("artifact_dir")
    require(isinstance(recorded_value, str) and Path(recorded_value).is_absolute(), "recorded artifact root invalid")
    recorded = Path(recorded_value)
    require(summary.get("pass_line") == f"BACKEND REGRESSION SMOKE PASS: {recorded}", "runner PASS line mismatch")
    rows = summary.get("scenarios")
    require(isinstance(rows, list) and len(rows) == 3, "summary scenarios missing")
    require([(row.get("name"), row.get("type")) for row in rows if isinstance(row, dict)] == list(SCENARIOS), "summary scenario order mismatch")
    for row in rows:
        require(isinstance(row, dict) and row.get("status") == "pass", "scenario result did not pass")
        artifact = resolve_member(source, recorded, row.get("artifact"), "summary scenario artifact")
        require(read_json(artifact) == row, f"summary scenario differs from {artifact}")

    receipt = read_json(source / "execution_receipt.json")
    validate_self_hash(receipt, "execution_receipt.json")
    require(receipt.get("schema_version") == 1 and receipt.get("mode") == "start", "execution mode mismatch")
    require(receipt.get("git_sha") == git_sha and receipt.get("dirty_status") == summary.get("dirty_status"), "execution source identity mismatch")
    require(receipt.get("backend") == "cuda" and receipt.get("model") == MODEL, "execution model/backend mismatch")
    require(receipt.get("selected_scenarios") == [name for name, _ in SCENARIOS], "execution scenario mismatch")
    require(receipt.get("scenario_count") == 3 and receipt.get("failed") == 0 and receipt.get("skipped") == 0, "execution result mismatch")
    execution_phases = receipt.get("scenario_execution_phases")
    require(isinstance(execution_phases, list) and len(execution_phases) == 2, "execution phase receipt mismatch")
    expected_phases = (
        ("run", [RUN_NAME]),
        ("serve", [SERVE_NAME, CONCURRENCY_NAME]),
    )
    parsed_phases: list[tuple[datetime, datetime]] = []
    for row, (expected_phase, expected_scenarios) in zip(
        execution_phases, expected_phases, strict=True
    ):
        require(
            isinstance(row, dict)
            and set(row) == {"phase", "scenarios", "started_at", "finished_at"},
            f"{expected_phase} execution phase shape mismatch",
        )
        require(
            row.get("phase") == expected_phase
            and row.get("scenarios") == expected_scenarios,
            f"{expected_phase} execution phase identity mismatch",
        )
        try:
            phase_started = datetime.fromisoformat(str(row.get("started_at")))
            phase_finished = datetime.fromisoformat(str(row.get("finished_at")))
        except ValueError as error:
            raise ValidationError(f"{expected_phase} execution phase timestamp invalid") from error
        require(
            phase_started.tzinfo is not None
            and phase_finished.tzinfo is not None
            and phase_started <= phase_finished,
            f"{expected_phase} execution phase interval invalid",
        )
        parsed_phases.append((phase_started, phase_finished))
    require(
        parsed_phases[0][1] <= parsed_phases[1][0],
        "run and serve execution phases overlap",
    )
    runner_argv = receipt.get("runner_argv")
    runner_path = receipt.get("runner_path")
    require(isinstance(runner_argv, list) and len(runner_argv) >= 2, "runner argv missing")
    require(isinstance(runner_path, str) and Path(runner_path).is_absolute(), "runner path invalid")
    require(Path(runner_argv[1]).resolve(strict=False) == Path(runner_path).resolve(strict=False), "runner argv/path mismatch")
    require(Path(runner_path).name == "run_scenarios.py", "runner identity mismatch")
    require("--manifest" in runner_argv and "--out" in runner_argv and "--only" not in runner_argv, "runner argv scope mismatch")
    cwd = Path(str(receipt.get("cwd") or ""))
    require(cwd.is_absolute(), "runner cwd invalid")
    manifest_arg = Path(runner_argv[runner_argv.index("--manifest") + 1])
    out_arg = Path(runner_argv[runner_argv.index("--out") + 1])
    if not manifest_arg.is_absolute():
        manifest_arg = cwd / manifest_arg
    if not out_arg.is_absolute():
        out_arg = cwd / out_arg
    require(manifest_arg.resolve(strict=False) == Path(str(receipt.get("manifest_path"))).resolve(strict=False), "runner manifest argv mismatch")
    require(out_arg.resolve(strict=False) == recorded.resolve(strict=False), "runner output argv mismatch")

    inputs = receipt.get("input_artifacts")
    require(isinstance(inputs, dict), "execution input artifacts missing")
    expected_inputs = {
        "runner": ("run_scenarios.py", RUNNER_PATH, receipt.get("runner_sha256")),
        "manifest": ("scenario_manifest.json", SCENARIO_MANIFEST_PATH, receipt.get("manifest_sha256")),
        "concurrency_quality_helper": (
            "openai_concurrency_quality_regression.py",
            HELPER_PATH,
            None,
        ),
    }
    require(set(inputs) == set(expected_inputs), "execution input artifact set mismatch")
    for key, (filename, current, receipt_sha) in expected_inputs.items():
        row = inputs[key]
        require(isinstance(row, dict) and set(row) == {"path", "sha256"}, f"execution input {key} invalid")
        path = resolve_member(source, recorded, row.get("path"), f"execution input {key}")
        require(path == source / "inputs" / filename, f"execution input {key} path mismatch")
        digest = file_sha256(path)
        require(row.get("sha256") == digest == file_sha256(current), f"execution input {key} SHA mismatch")
        if receipt_sha is not None:
            require(receipt_sha == digest, f"execution receipt {key} SHA mismatch")

    binary_path = receipt.get("binary_path")
    binary_sha = receipt.get("binary_sha256")
    require(isinstance(binary_path, str) and Path(binary_path).is_absolute(), "binary path invalid")
    require(isinstance(binary_sha, str) and SHA256_RE.fullmatch(binary_sha), "binary SHA invalid")
    server_argv = receipt.get("server_argv")
    require(isinstance(server_argv, list) and server_argv and server_argv[0] == binary_path, "server argv/binary mismatch")
    require("serve" in server_argv and MODEL in server_argv, "server argv product identity missing")
    require(any(server_argv[i : i + 2] == ["--backend", "cuda"] for i in range(len(server_argv) - 1)), "server argv lacks typed CUDA backend")
    require(
        any(
            server_argv[i : i + 2] == ["--max-num-seqs", "1"]
            for i in range(len(server_argv) - 1)
        ),
        "server argv must align typed capacity with run",
    )
    hardware = receipt.get("hardware")
    require(isinstance(hardware, dict) and hardware.get("returncode") == 0, "hardware probe failed")
    gpu_rows = [row for row in str(hardware.get("stdout") or "").splitlines() if row.strip()]
    require(len(gpu_rows) == 1 and "RTX 4090" in gpu_rows[0], "hardware must prove exactly one RTX 4090")
    removed = receipt.get("removed_hidden_env_names")
    require(isinstance(removed, list) and all(isinstance(key, str) and key.startswith("FERRUM_") for key in removed), "removed hidden env receipt invalid")
    child_env = receipt.get("child_env")
    require(isinstance(child_env, dict) and not any(str(key).startswith("FERRUM_") for key in child_env), "hidden FERRUM env reached child")
    require(child_env.get("NO_COLOR") == "1", "server child env omitted NO_COLOR policy")
    require(receipt.get("server_returncode") in (0, -15), "server return code mismatch")
    for field in ("server_started_at", "server_finished_at"):
        require(isinstance(receipt.get(field), str) and receipt[field], f"{field} missing")
    try:
        server_started_at = datetime.fromisoformat(receipt["server_started_at"])
    except ValueError as error:
        raise ValidationError("server_started_at invalid") from error
    require(
        parsed_phases[0][1] <= parsed_phases[1][0] <= server_started_at <= parsed_phases[1][1],
        "server did not start strictly after the run-only phase",
    )
    evidence = receipt.get("evidence_files")
    expected_evidence = {
        "effective_config": "server.effective_config.json",
        "decision_trace": "server.decision_trace.jsonl",
        "server_log": "server.log",
        "health_before": "server.health.json",
        "health_after": "server.health.after.json",
    }
    require(isinstance(evidence, dict) and set(evidence) == set(expected_evidence), "execution evidence set mismatch")
    for key, filename in expected_evidence.items():
        row = evidence[key]
        require(isinstance(row, dict), f"execution evidence {key} missing")
        path = resolve_member(source, recorded, row.get("path"), f"execution evidence {key}")
        require(path == source / filename, f"execution evidence {key} path mismatch")
        require(row.get("size") == path.stat().st_size and row.get("sha256") == file_sha256(path), f"execution evidence {key} binding mismatch")
    effective = read_json(source / "server.effective_config.json")
    require(effective.get("backend") == "cuda" and effective.get("cuda_device_count") == 1, "effective CUDA config mismatch")
    require(effective.get("selected_gpu_devices") == [0], "effective CUDA device mismatch")
    require(
        effective.get("selected_max_sequences") == 1
        and effective.get("selected_admission_limit") == 1,
        "effective run/serve capacity alignment mismatch",
    )
    require(effective.get("model_capabilities", {}).get("architecture") == "qwen3_5", "effective architecture mismatch")
    require(effective.get("hardware_capabilities", {}).get("backend") == "cuda", "effective hardware backend mismatch")
    require(effective.get("hardware_capabilities", {}).get("compiled_features", {}).get("cuda") is True, "effective CUDA feature missing")
    for filename in ("server.health.json", "server.health.after.json"):
        health = read_json(source / filename)
        require(health.get("status") == "pass" and health.get("http_status") == 200, f"{filename}: health mismatch")
    assert_clean_output("server.log", read_text(source / "server.log"))
    assert_clean_output("server.decision_trace.jsonl", read_text(source / "server.decision_trace.jsonl"))
    require(
        hardware.get("argv")
        == [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        "hardware probe command mismatch",
    )
    summary_receipt = summary.get("execution_receipt")
    require(isinstance(summary_receipt, dict), "summary execution receipt missing")
    summary_receipt_path = resolve_member(
        source,
        recorded,
        summary_receipt.get("artifact"),
        "summary execution receipt",
    )
    require(summary_receipt_path == source / "execution_receipt.json", "summary execution receipt path mismatch")
    require(summary_receipt.get("artifact_sha256") == file_sha256(summary_receipt_path), "summary execution receipt file SHA mismatch")
    require(summary_receipt.get("canonical_sha256") == receipt.get("canonical_sha256"), "summary execution receipt canonical SHA mismatch")
    require(summary_receipt.get("mode") == "start", "summary execution mode mismatch")
    require(summary_receipt.get("runner_sha256") == receipt.get("runner_sha256"), "summary runner SHA mismatch")
    require(summary_receipt.get("manifest_sha256") == receipt.get("manifest_sha256"), "summary manifest SHA mismatch")
    require(summary_receipt.get("binary_sha256") == binary_sha, "summary binary SHA mismatch")
    validate_manifest(source)
    return summary, recorded, receipt


def validate_run(source: Path, recorded: Path, receipt: dict[str, Any], row: dict[str, Any]) -> tuple[set[str], dict[str, Any]]:
    root = source / RUN_NAME
    require(
        row.get("assistant_turns") == len(RUN_PROMPTS)
        and row.get("length_finishes") == 0
        and row.get("used_default_max_tokens") is True,
        "run multi-turn result mismatch",
    )
    expected_input = "\n".join(RUN_PROMPTS) + "\n/bye\n"
    input_text = read_text(root / "input.txt")
    require(input_text == expected_input, "run multi-turn stdin mismatch")
    command = read_json(root / "command.json")
    argv = command.get("argv")
    require(isinstance(argv, list) and argv and argv[0] == receipt.get("binary_path"), "run command binary mismatch")
    require(command.get("binary_sha256") == receipt.get("binary_sha256"), "run/server binary SHA mismatch")
    require(command.get("stdin_sha256") == hashlib.sha256(input_text.encode()).hexdigest(), "run stdin SHA mismatch")
    stdin_path = resolve_member(source, recorded, command.get("stdin_path"), "run stdin path")
    require(stdin_path == root / "input.txt", "run stdin path mismatch")
    for flag, expected in (("--backend", "cuda"), ("--temperature", "0.0"), ("--output-format", "jsonl")):
        require(flag in argv and argv[argv.index(flag) + 1] == expected, f"run command {flag} mismatch")
    require("--max-tokens" not in argv, "H02.1 run command must use the product default max-token budget")
    require("--disable-thinking" in argv and "--enable-thinking" not in argv, "run thinking mode mismatch")
    require(argv[-1] == MODEL and "run" in argv, "run product identity mismatch")
    require(isinstance(command.get("cwd"), str) and Path(command["cwd"]).is_absolute(), "run cwd invalid")
    require(command.get("cwd") == receipt.get("cwd"), "run/server cwd receipts differ")
    require(command.get("env_policy") == "remove_FERRUM_prefix", "run env policy mismatch")
    run_removed = command.get("removed_hidden_env_names")
    run_env = command.get("child_env")
    require(
        isinstance(run_removed, list)
        and all(isinstance(key, str) and key.startswith("FERRUM_") for key in run_removed)
        and len(run_removed) == len(set(run_removed)),
        "run removed hidden env receipt invalid",
    )
    require(
        isinstance(run_env, dict)
        and run_env.get("NO_COLOR") == "1"
        and not any(str(key).startswith("FERRUM_") for key in run_env),
        "hidden FERRUM env reached run child",
    )
    require(run_removed == receipt.get("removed_hidden_env_names"), "run/server removed env receipts differ")
    require(run_env == receipt.get("child_env"), "run/server child env receipts differ")
    for flag, relative in (
        ("--effective-config-json", "effective_config.json"),
        ("--decision-trace-jsonl", "decision_trace.jsonl"),
        ("--scheduler-trace-jsonl", "observability/scheduler_trace.jsonl"),
        ("--request-dump-dir", "observability/request_dump"),
    ):
        require(flag in argv and argv.index(flag) + 1 < len(argv), f"run command omits {flag}")
        value = Path(argv[argv.index(flag) + 1])
        require(value == recorded / RUN_NAME / relative, f"run command {flag} path mismatch")
    assert_clean_output("run stderr", read_text(root / "stderr.log"))
    events = read_jsonl(root / "stdout.jsonl")
    ready = [event for event in events if event.get("event") == "ready"]
    exits = [event for event in events if event.get("event") == "exit"]
    users = [event for event in events if event.get("event") == "user"]
    assistants = [event for event in events if event.get("event") == "assistant"]
    require(len(ready) == 1 and ready[0].get("backend") == "CUDA(0)", "run ready event mismatch")
    require(len(exits) == 1, "run exit event mismatch")
    require(
        len(users) == len(RUN_PROMPTS) and len(assistants) == len(RUN_PROMPTS),
        "run turn cardinality mismatch",
    )
    session_ids = {event.get("session_id") for event in [*users, *assistants]}
    require(len(session_ids) == 1 and None not in session_ids, "run session identity mismatch")
    expected_turns = list(range(len(RUN_PROMPTS)))
    require([event.get("turn") for event in users] == expected_turns, "run user turn order mismatch")
    require([event.get("turn") for event in assistants] == expected_turns, "run assistant turn order mismatch")
    request_ids: set[str] = set()
    for turn, expected_prompt in enumerate(RUN_PROMPTS):
        user = users[turn]
        assistant = assistants[turn]
        request_id = user.get("request_id")
        require(isinstance(request_id, str) and request_id, f"run turn {turn}: request id missing")
        require(assistant.get("request_id") == request_id, f"run turn {turn}: request/response id mismatch")
        require(user.get("content") == expected_prompt, f"run turn {turn}: prompt mismatch")
        request_ids.add(request_id)
    require(len(request_ids) == len(RUN_PROMPTS), "run request ids are not unique")
    for event in assistants:
        require(event.get("finish_reason") in {"stop", "eos"}, "run assistant finish mismatch")
        require(
            event.get("reasoning") in (None, "")
            and event.get("reasoning_content") in (None, ""),
            "run thinking was not disabled",
        )
        validate_usage(event.get("usage"), "run assistant usage")
        assert_clean_output("run assistant", str(event.get("content") or ""))
    for turn, expected_answer in enumerate(RUN_ANSWERS):
        require_exact_visible_answer(
            assistants[turn].get("content"), expected_answer, f"run turn {turn}"
        )
    effective = read_json(root / "effective_config.json")
    require(effective.get("backend") == "cuda", "run effective backend mismatch")
    require(
        effective.get("selected_max_sequences") == 1
        and effective.get("selected_admission_limit") == 1,
        "run effective capacity no longer matches the single-sequence product default",
    )
    read_jsonl(root / "decision_trace.jsonl")
    return request_ids, {
        "assistant_turns": len(RUN_PROMPTS),
        "session_id": next(iter(session_ids)),
        "historical_case_id": HISTORICAL_CASE_ID,
        "used_default_max_tokens": True,
    }


def validate_serve_multiturn(source: Path, row: dict[str, Any]) -> dict[str, Any]:
    root = source / SERVE_NAME
    require(row.get("assistant_turns") == 2 and row.get("recalled_secret") is True, "serve multi-turn result mismatch")
    first_request = read_json(root / "turn1.request.json")
    second_request = read_json(root / "turn2.request.json")
    expected_first = "Remember the codeword banana. Reply exactly OK."
    expected_second = "What codeword did I ask you to remember? Answer with only the codeword."
    require(first_request.get("model") == MODEL and first_request.get("temperature") == 0.0, "serve turn1 request mismatch")
    require(first_request.get("max_tokens") == 128 and first_request.get("chat_template_kwargs") == {"enable_thinking": False}, "serve turn1 typed options mismatch")
    require(first_request.get("messages") == [{"role": "user", "content": expected_first}], "serve turn1 messages mismatch")
    first = validate_chat_response(root / "turn1.json", expected="OK")
    expected_messages = [
        {"role": "user", "content": expected_first},
        {"role": "assistant", "content": first["content"]},
        {"role": "user", "content": expected_second},
    ]
    require(second_request.get("model") == MODEL and second_request.get("temperature") == 0.0, "serve turn2 request mismatch")
    require(second_request.get("max_tokens") == 128 and second_request.get("chat_template_kwargs") == {"enable_thinking": False}, "serve turn2 typed options mismatch")
    require(second_request.get("messages") == expected_messages, "serve turn2 did not carry conversation history")
    second = validate_chat_response(root / "turn2.json", expected=SECRET)
    require(first["id"] != second["id"], "serve turns reused response id")
    return {"assistant_turns": 2, "response_ids": [first["id"], second["id"]]}


def expected_concurrency_request(concurrency: int, index: int) -> dict[str, Any]:
    marker = f"ferrum{concurrency:02d}{index:02d}"
    checksum = f"S{(index + 1) ** 2:04d}"
    return {
        "model": MODEL,
        "temperature": 0.0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "capture_quality_marker",
                    "description": "Record one concurrency quality marker.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "marker": {"type": "string", "enum": [marker]},
                            "checksum": {"type": "string", "enum": [checksum]},
                        },
                        "required": ["marker", "checksum"],
                    },
                },
            }
        ],
        "tool_choice": "required",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Call capture_quality_marker with marker "
                    f"{marker!r} and checksum {checksum!r}. "
                    "Do not output natural language."
                ),
            }
        ],
    }


def validate_concurrency_response(
    path: Path,
    marker: str,
    checksum: str,
    all_markers: set[str],
) -> dict[str, Any]:
    raw = read_text(path)
    assert_clean_output(str(path), raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValidationError(f"{path}: malformed response JSON: {error}") from error
    require(
        isinstance(value, dict)
        and value.get("object") == "chat.completion"
        and value.get("model") == MODEL,
        f"{path}: response object/model mismatch",
    )
    response_id = value.get("id")
    require(isinstance(response_id, str) and response_id, f"{path}: response id missing")
    choices = value.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, f"{path}: choices mismatch")
    choice = choices[0]
    require(
        isinstance(choice, dict)
        and choice.get("index") == 0
        and choice.get("finish_reason") == "tool_calls",
        f"{path}: choice/finish reason mismatch",
    )
    message = choice.get("message")
    require(isinstance(message, dict), f"{path}: message missing")
    require(message.get("role") == "assistant", f"{path}: message role mismatch")
    require(message.get("content") in (None, ""), f"{path}: natural-language content leaked")
    require(
        message.get("reasoning") in (None, "")
        and message.get("reasoning_content") in (None, ""),
        f"{path}: reasoning content leaked",
    )
    calls = message.get("tool_calls")
    require(isinstance(calls, list) and len(calls) == 1, f"{path}: tool-call cardinality mismatch")
    call = calls[0]
    require(isinstance(call, dict) and call.get("type") == "function", f"{path}: tool-call type mismatch")
    function = call.get("function")
    require(isinstance(function, dict) and function.get("name") == "capture_quality_marker", f"{path}: tool name mismatch")
    arguments = function.get("arguments")
    require(isinstance(arguments, str), f"{path}: tool arguments missing")
    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError as error:
        raise ValidationError(f"{path}: malformed tool arguments: {error}") from error
    require(parsed_arguments == {"marker": marker, "checksum": checksum}, f"{path}: tool arguments mismatch")
    validate_usage(value.get("usage"), str(path))
    leaked = sorted(candidate for candidate in all_markers - {marker} if candidate in raw)
    require(not leaked, f"{path}: cross-request markers leaked: {leaked}")
    return {"response_id": response_id, "raw": raw}


def validate_concurrency(source: Path, recorded: Path, row: dict[str, Any]) -> dict[str, Any]:
    root = source / CONCURRENCY_NAME
    aggregate = read_json(root / "concurrency_quality_regression.json")
    require(aggregate.get("status") == "pass" and aggregate.get("model") == MODEL, "concurrency aggregate mismatch")
    cells = aggregate.get("cells")
    require(isinstance(cells, list) and [cell.get("concurrency") for cell in cells if isinstance(cell, dict)] == list(EXPECTED_CONCURRENCY), "concurrency aggregate cells mismatch")
    require(row.get("cells") == cells, "scenario result concurrency cells mismatch")
    all_markers = {
        f"ferrum{concurrency:02d}{index:02d}"
        for concurrency in EXPECTED_CONCURRENCY
        for index in range(concurrency)
    }
    observed: dict[int, dict[str, Any]] = {}
    response_ids: set[str] = set()
    for concurrency in EXPECTED_CONCURRENCY:
        cell = read_json(root / f"c{concurrency}.quality.json")
        require(cell.get("concurrency") == concurrency and cell.get("requests") == concurrency, f"c{concurrency}: request count mismatch")
        require(cell.get("worker_limit") == WORKER_LIMIT, f"c{concurrency}: independent worker limit missing")
        require(cell.get("worker_count") == min(concurrency, WORKER_LIMIT), f"c{concurrency}: worker count mismatch")
        require(
            cell.get("synchronized_start") is (concurrency > 1),
            f"c{concurrency}: synchronized-start receipt mismatch",
        )
        for key in ("status_200", "json_ok", "marker_ok", "square_ok", "format_ok"):
            require(cell.get(key) == concurrency, f"c{concurrency}: {key} mismatch")
        for key in ("crosstalk", "length_finishes", "forbidden_count"):
            require(cell.get(key) == 0, f"c{concurrency}: {key} is not zero")
        require(cell.get("passed") is True, f"c{concurrency}: passed is false")
        rows = cell.get("rows")
        require(isinstance(rows, list) and len(rows) == concurrency, f"c{concurrency}: row count mismatch")
        expected_summary = {key: value for key, value in cell.items() if key != "rows"}
        require(cells[EXPECTED_CONCURRENCY.index(concurrency)] == expected_summary, f"c{concurrency}: aggregate binding mismatch")
        markers: set[str] = set()
        intervals: list[tuple[int, int]] = []
        for index, item in enumerate(rows):
            require(isinstance(item, dict) and item.get("i") == index, f"c{concurrency}: row order mismatch")
            marker = f"ferrum{concurrency:02d}{index:02d}"
            checksum = f"S{(index + 1) ** 2:04d}"
            request_path = resolve_member(
                source,
                recorded,
                item.get("request_artifact"),
                f"c{concurrency}/{index} request artifact",
            )
            response_path = resolve_member(
                source,
                recorded,
                item.get("response_artifact"),
                f"c{concurrency}/{index} response artifact",
            )
            require(
                request_path
                == root / f"c{concurrency}.quality.{index:03d}.request.json",
                f"c{concurrency}/{index}: request path mismatch",
            )
            require(
                response_path
                == root / f"c{concurrency}.quality.{index:03d}.response.txt",
                f"c{concurrency}/{index}: response path mismatch",
            )
            require(
                item.get("request_sha256") == file_sha256(request_path),
                f"c{concurrency}/{index}: request SHA mismatch",
            )
            require(
                item.get("response_sha256") == file_sha256(response_path),
                f"c{concurrency}/{index}: response SHA mismatch",
            )
            require(
                item.get("response_size") == response_path.stat().st_size > 0,
                f"c{concurrency}/{index}: response size mismatch",
            )
            require(
                read_json(request_path) == expected_concurrency_request(concurrency, index),
                f"c{concurrency}/{index}: raw request mismatch",
            )
            raw_response = validate_concurrency_response(
                response_path,
                marker,
                checksum,
                all_markers,
            )
            require(
                item.get("content_head") == raw_response["raw"][:500],
                f"c{concurrency}/{index}: response diagnostic is not raw-bound",
            )
            require(
                raw_response["response_id"] not in response_ids,
                f"c{concurrency}/{index}: duplicate response id",
            )
            response_ids.add(raw_response["response_id"])
            require(item.get("status") == 200 and item.get("json_ok") is True, f"c{concurrency}/{index}: response mismatch")
            require(item.get("marker") == marker == item.get("parsed_marker"), f"c{concurrency}/{index}: marker mismatch")
            require(item.get("square") == checksum == item.get("parsed_checksum"), f"c{concurrency}/{index}: checksum mismatch")
            require(item.get("tool_name") == "capture_quality_marker" and item.get("format_ok") is True, f"c{concurrency}/{index}: tool format mismatch")
            require(item.get("finish_reason") == "tool_calls" and item.get("forbidden_text") is None, f"c{concurrency}/{index}: finish/output mismatch")
            assert_clean_output(f"c{concurrency}/{index}", str(item.get("content_head") or ""))
            started = item.get("started_monotonic_ns")
            finished = item.get("finished_monotonic_ns")
            duration_ms = item.get("duration_ms")
            require(
                isinstance(started, int)
                and not isinstance(started, bool)
                and isinstance(finished, int)
                and not isinstance(finished, bool)
                and 0 < started < finished,
                f"c{concurrency}/{index}: invalid monotonic interval",
            )
            require(
                isinstance(duration_ms, (int, float))
                and not isinstance(duration_ms, bool)
                and duration_ms > 0
                and abs(float(duration_ms) - (finished - started) / 1_000_000) <= 0.001,
                f"c{concurrency}/{index}: duration is not interval-bound",
            )
            intervals.append((started, finished))
            markers.add(marker)
        require(len(markers) == concurrency, f"c{concurrency}: marker uniqueness mismatch")
        overlap_pairs = sum(
            1
            for left_index, left in enumerate(intervals)
            for right in intervals[left_index + 1 :]
            if max(left[0], right[0]) < min(left[1], right[1])
        )
        timeline = sorted(
            [(started, 1) for started, _ in intervals]
            + [(finished, -1) for _, finished in intervals],
            key=lambda event: (event[0], event[1]),
        )
        active = 0
        max_in_flight = 0
        for _, delta in timeline:
            active += delta
            max_in_flight = max(max_in_flight, active)
        require(cell.get("overlap_pair_count") == overlap_pairs, f"c{concurrency}: overlap count mismatch")
        require(cell.get("max_in_flight") == max_in_flight, f"c{concurrency}: max-in-flight mismatch")
        if concurrency == 1:
            require(overlap_pairs == 0 and max_in_flight == 1, "c1 overlap baseline mismatch")
        else:
            require(overlap_pairs >= 1 and max_in_flight >= 2, f"c{concurrency}: requests never overlapped")
        observed[concurrency] = {
            "requests": concurrency,
            "worker_count": cell["worker_count"],
            "overlap_pair_count": overlap_pairs,
            "max_in_flight": max_in_flight,
        }
    require(len(response_ids) == sum(EXPECTED_CONCURRENCY), "concurrency response id count mismatch")
    return {
        "cells": observed,
        "request_count": sum(EXPECTED_CONCURRENCY),
        "response_id_count": len(response_ids),
    }


def validate_lifecycle(
    rows: list[dict[str, Any]], request_id: str, entrypoint: str
) -> dict[str, Any]:
    resource_rows = [row for row in rows if row.get("request_id") == request_id]
    require(resource_rows, f"request has no resource trace: {request_id}")
    closes = [row for row in resource_rows if row.get("phase") == "engine_request_close"]
    require(len(closes) == 1, f"request close cardinality mismatch: {request_id}")
    close = closes[0]
    shape = close.get("shape") if isinstance(close.get("shape"), dict) else {}
    attrs = close.get("attributes") if isinstance(close.get("attributes"), dict) else {}
    require(shape.get("resource_owner_outstanding_count", attrs.get("resource_owner_outstanding_count")) == 0, f"request resources outstanding: {request_id}")
    validate_resource_balance(rows, request_id)
    execution_id = f"request.product.{request_id}"
    execution = [row for row in rows if row.get("request_id") == execution_id]
    require(EXECUTION_PHASES <= {str(row.get("phase")) for row in execution}, f"vNext lifecycle incomplete: {request_id}")
    plan_hashes: set[str] = set()
    runtime_fingerprints: set[str] = set()
    legacy_selection_count = 0
    for event in execution:
        if event.get("phase") not in EXECUTION_PHASES:
            continue
        require(event.get("entrypoint") == entrypoint and event.get("backend") == "actual" and event.get("status") == "ok", f"vNext provenance mismatch: {request_id}")
        detail = event.get("backend_detail")
        attributes = event.get("attributes")
        require(isinstance(detail, dict) and detail.get("backend_device") == "CUDA(0)", f"vNext device mismatch: {request_id}")
        require(
            isinstance(attributes, dict)
            and attributes.get("execution_trace_source") == "vnext"
            and attributes.get("actual_model_smoke") is True
            and attributes.get("diagnostic_only") is False
            and attributes.get("l0_only") is False,
            f"vNext source mismatch: {request_id}",
        )
        if attributes.get("execution_trace_source") != "vnext":
            legacy_selection_count += 1
        plan_hash = attributes.get("plan_hash")
        runtime_fingerprint = attributes.get("runtime_implementation_fingerprint")
        if plan_hash is not None:
            require(
                isinstance(plan_hash, str) and SHA256_RE.fullmatch(plan_hash),
                f"vNext plan hash is invalid: {request_id}",
            )
            plan_hashes.add(plan_hash)
        if runtime_fingerprint is not None:
            require(
                isinstance(runtime_fingerprint, str)
                and SHA256_RE.fullmatch(runtime_fingerprint),
                f"vNext runtime fingerprint is invalid: {request_id}",
            )
            runtime_fingerprints.add(runtime_fingerprint)
        if event.get("phase") == "vnext.operation_submitted":
            require(str(attributes.get("provider_id") or "").startswith("provider.cuda."), f"CUDA provider missing: {request_id}")
            require(attributes.get("device_id") == "device.cuda.0", f"CUDA device id missing: {request_id}")
    require(len(plan_hashes) == 1, f"vNext request plan identity is missing or ambiguous: {request_id}")
    require(
        len(runtime_fingerprints) == 1,
        f"vNext request runtime identity is missing or ambiguous: {request_id}",
    )
    require(legacy_selection_count == 0, f"legacy execution selected: {request_id}")
    return {
        "entrypoint": entrypoint,
        "resolved_execution_plan_hash": next(iter(plan_hashes)),
        "runtime_implementation_fingerprint": next(iter(runtime_fingerprints)),
        "legacy_selection_count": 0,
    }


def request_bundles(root: Path, *, expected_startup: int) -> tuple[list[Path], list[Path]]:
    bundles = sorted(path for path in root.iterdir() if path.is_dir() and not path.is_symlink())
    startup = [path for path in bundles if path.name.startswith("serve-startup-") or path.name.startswith("run-startup-")]
    product = [path for path in bundles if path not in startup]
    require(len(startup) == expected_startup, f"startup request bundle count mismatch under {root}")
    return startup, product


def validate_bundle(bundle: Path, entrypoint: str) -> dict[str, Any]:
    request = read_json(bundle / "request.json")
    require(request.get("request_id") == bundle.name, f"request dump identity mismatch: {bundle}")
    require(request.get("entrypoint") == entrypoint and request.get("backend") == "actual", f"request dump provenance mismatch: {bundle}")
    require(is_qwen35_model(request.get("model")) and request.get("actual_model_smoke") is True and request.get("sanitized") is True, f"request dump model mismatch: {bundle}")
    backend = read_json(bundle / "backend_selection.json")
    require(
        backend.get("request_id") == bundle.name
        and backend.get("backend") == "actual"
        and is_qwen35_model(backend.get("model")),
        f"backend selection mismatch: {bundle}",
    )
    bad = read_json(bundle / "bad_output_scan.json")
    require(bad.get("request_id") == bundle.name and bad.get("bad_output") is False and bad.get("bad_text_count") == 0 and bad.get("reasons") == [], f"bad output scan failed: {bundle}")
    return request


def validate_observability(source: Path, recorded: Path, summary: dict[str, Any], run_ids: set[str]) -> dict[str, Any]:
    obs = summary.get("observability")
    require(isinstance(obs, dict) and obs.get("enabled") is True, "observability summary missing")
    require(read_json(source / "observability_summary.json") == obs, "observability summary artifact mismatch")
    run_root = source / RUN_NAME / "observability"
    serve_root = source / "observability/serve"
    scheduler_paths = obs.get("scheduler_trace_paths")
    dump_paths = obs.get("request_dump_dirs")
    require(isinstance(scheduler_paths, list) and len(scheduler_paths) == 2, "scheduler trace path count mismatch")
    require(isinstance(dump_paths, list) and len(dump_paths) == 2, "request dump path count mismatch")
    resolved_scheduler = {resolve_member(source, recorded, value, "scheduler trace") for value in scheduler_paths}
    resolved_dumps = {resolve_member(source, recorded, value, "request dump") for value in dump_paths}
    require(resolved_scheduler == {run_root / "scheduler_trace.jsonl", serve_root / "scheduler_trace.jsonl"}, "scheduler trace roots mismatch")
    require(resolved_dumps == {run_root / "request_dump", serve_root / "request_dump"}, "request dump roots mismatch")

    run_rows = read_jsonl(run_root / "scheduler_trace.jsonl")
    _, run_bundles = request_bundles(run_root / "request_dump", expected_startup=0)
    require(len(run_bundles) == EXPECTED_RUN_REQUESTS, "run request dump count mismatch")
    require({bundle.name for bundle in run_bundles} == run_ids, "run stdout/request-dump identity mismatch")
    execution_identities: dict[str, list[dict[str, Any]]] = {"run": [], "serve": []}
    for bundle in run_bundles:
        request = validate_bundle(bundle, "run")
        require("http" not in request, "run request dump masquerades as HTTP")
        execution_identities["run"].append(
            validate_lifecycle(run_rows, bundle.name, "run")
        )

    serve_rows = read_jsonl(serve_root / "scheduler_trace.jsonl")
    _, serve_bundles = request_bundles(serve_root / "request_dump", expected_startup=1)
    require(len(serve_bundles) == EXPECTED_SERVE_REQUESTS, "serve request dump count mismatch")
    expected_bodies: dict[str, str] = {}

    def register_expected_body(label: str, value: dict[str, Any]) -> None:
        canonical = json.dumps(redacted_request_body(value), ensure_ascii=False, sort_keys=True)
        require(canonical not in expected_bodies, f"duplicate expected request body: {label}")
        expected_bodies[canonical] = label

    register_expected_body("turn1", read_json(source / SERVE_NAME / "turn1.request.json"))
    register_expected_body("turn2", read_json(source / SERVE_NAME / "turn2.request.json"))
    for concurrency in EXPECTED_CONCURRENCY:
        for index in range(concurrency):
            register_expected_body(
                f"concurrency:{concurrency}:{index}",
                read_json(
                    source
                    / CONCURRENCY_NAME
                    / f"c{concurrency}.quality.{index:03d}.request.json"
                ),
            )

    categories: list[str] = []
    markers: set[str] = set()
    observed_bodies: set[str] = set()
    for bundle in serve_bundles:
        request = validate_bundle(bundle, "serve")
        http = request.get("http")
        require(isinstance(http, dict) and http.get("method") == "POST" and http.get("path") == "/v1/chat/completions", f"HTTP request dump mismatch: {bundle}")
        body = http.get("body")
        require(isinstance(body, dict), f"HTTP body missing: {bundle}")
        canonical = json.dumps(body, ensure_ascii=False, sort_keys=True)
        require(canonical in expected_bodies, f"request dump body differs from every raw request: {bundle}")
        require(canonical not in observed_bodies, f"duplicate request dump body: {bundle}")
        observed_bodies.add(canonical)
        label = expected_bodies[canonical]
        if label.startswith("concurrency:"):
            _, concurrency_text, index_text = label.split(":")
            marker = f"ferrum{int(concurrency_text):02d}{int(index_text):02d}"
            markers.add(marker)
            categories.append("concurrency")
        else:
            categories.append(label)
        execution_identities["serve"].append(
            validate_lifecycle(serve_rows, bundle.name, "serve")
        )
    require(observed_bodies == set(expected_bodies), "serve request dump/body matrix incomplete")
    require(categories.count("turn1") == 1 and categories.count("turn2") == 1, "serve multi-turn request matrix incomplete")
    require(categories.count("concurrency") == 5, "serve concurrency request count mismatch")
    expected_markers = {"ferrum0100", "ferrum0400", "ferrum0401", "ferrum0402", "ferrum0403"}
    require(markers == expected_markers, "serve concurrency request markers incomplete")
    plan_hashes = {
        row["resolved_execution_plan_hash"]
        for rows in execution_identities.values()
        for row in rows
    }
    runtime_fingerprints = {
        row["runtime_implementation_fingerprint"]
        for rows in execution_identities.values()
        for row in rows
    }
    legacy_selection_count = sum(
        row["legacy_selection_count"]
        for rows in execution_identities.values()
        for row in rows
    )
    require(len(plan_hashes) == 1, "run and serve resolved execution plans differ")
    require(len(runtime_fingerprints) == 1, "run and serve runtime implementations differ")
    require(legacy_selection_count == 0, "run or serve selected legacy execution")
    return {
        "run_requests": EXPECTED_RUN_REQUESTS,
        "serve_requests": EXPECTED_SERVE_REQUESTS,
        "execution_lifecycles": EXPECTED_RUN_REQUESTS + EXPECTED_SERVE_REQUESTS,
        "product_execution_identity": {
            "entrypoints": ["run", "serve"],
            "resolved_execution_plan_hash": next(iter(plan_hashes)),
            "runtime_implementation_fingerprint": next(iter(runtime_fingerprints)),
            "same_resolved_execution_plan": True,
            "same_runtime_implementation": True,
            "production_legacy_selection_count": 0,
        },
    }


def validate_source(source: Path, expected_git_sha: str | None = None) -> dict[str, Any]:
    source = source.resolve(strict=True)
    summary, recorded, receipt = validate_identity(source, expected_git_sha)
    rows = summary["scenarios"]
    run_ids, run = validate_run(source, recorded, receipt, rows[0])
    serve = validate_serve_multiturn(source, rows[1])
    concurrency = validate_concurrency(source, recorded, rows[2])
    observability = validate_observability(source, recorded, summary, run_ids)
    tree = validate_tree(source, recorded)
    return {
        "git_sha": summary["git_sha"],
        "binary_sha256": receipt["binary_sha256"],
        "backend": "cuda",
        "model": MODEL,
        "scope": ["S2/run-multiturn", "S2/serve-multiturn", "S2/c1-c4-concurrency"],
        "full_s2": False,
        "does_not_prove": DOES_NOT_PROVE,
        "run": run,
        "serve": serve,
        "concurrency": concurrency,
        "observability": observability,
        "artifact_tree": tree,
    }


def validate_output_layout(source: Path, out: Path) -> tuple[Path, Path]:
    source = source.resolve(strict=True)
    out = out.resolve(strict=False)
    require(out != source, "checkpoint output must differ from source artifact root")
    require(
        source not in out.parents,
        "checkpoint output must not be inside source artifact root",
    )
    return source, out


def run_checkpoint(source: Path, out: Path, expected_git_sha: str | None) -> int:
    started_at = iso_now()
    started = time.monotonic()
    try:
        source, out = validate_output_layout(source, out)
    except (OSError, ValidationError) as error:
        print(f"{FAIL_PREFIX}: {out.resolve(strict=False)}: {error}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_id": CHECKPOINT_ID,
        "scope": ["S2/run-multiturn", "S2/serve-multiturn", "S2/c1-c4-concurrency"],
        "full_s2": False,
        "does_not_prove": DOES_NOT_PROVE,
        "source_root": str(source),
        "artifact_dir": str(out),
        "started_at": started_at,
    }
    try:
        evidence = validate_source(source, expected_git_sha)
    except (OSError, ValidationError) as error:
        manifest.update({"status": "fail", "finished_at": iso_now(), "duration_sec": time.monotonic() - started, "evidence": None, "error": str(error), "pass_line": None})
        write_json(out / "manifest.json", manifest)
        print(f"{FAIL_PREFIX}: {out}: {error}", file=sys.stderr)
        return 1
    pass_line = f"{PASS_PREFIX}: {out}"
    manifest.update({"status": "pass", "finished_at": iso_now(), "duration_sec": time.monotonic() - started, "evidence": evidence, "error": None, "pass_line": pass_line})
    write_json(out / "manifest.json", manifest)
    print(pass_line)
    return 0


def self_hash(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["canonical_sha256_scope"] = "document_without_canonical_sha256_fields"
    result["canonical_sha256"] = json_sha256(result)
    return result


def lifecycle_rows(request_id: str, entrypoint: str) -> list[dict[str, Any]]:
    resource = {"owner_kind": "request", "owner_id": request_id, "resource_kind": "request_slot"}
    rows = [
        {"request_id": request_id, "phase": "engine_request_open", "resource": {**resource, "action": "request_open"}},
        {"request_id": request_id, "phase": "engine_request_slot_reserve", "resource": {**resource, "action": "reserve", "amount": 1, "before": 0, "after": 1}},
        {"request_id": request_id, "phase": "engine_request_slot_commit", "resource": {**resource, "action": "commit", "amount": 1, "before": 0, "after": 1}},
        {"request_id": request_id, "phase": "engine_request_slot_release", "resource": {**resource, "action": "release", "amount": 1, "before": 1, "after": 0}},
        {"request_id": request_id, "phase": "engine_request_close", "attributes": {"resource_owner_outstanding_count": 0}, "shape": {"resource_owner_outstanding_count": 0}, "resource": {**resource, "action": "request_close"}},
    ]
    for phase in sorted(EXECUTION_PHASES):
        attributes = {
            "execution_trace_source": "vnext",
            "actual_model_smoke": True,
            "diagnostic_only": False,
            "l0_only": False,
            "plan_hash": "1" * 64,
            "runtime_implementation_fingerprint": "2" * 64,
        }
        if phase == "vnext.operation_submitted":
            attributes.update({"provider_id": "provider.cuda.fixture", "device_id": "device.cuda.0"})
        rows.append({
            "request_id": f"request.product.{request_id}",
            "phase": phase,
            "entrypoint": entrypoint,
            "backend": "actual",
            "status": "ok",
            "backend_detail": {"backend_device": "CUDA(0)"},
            "attributes": attributes,
        })
    return rows


def write_bundle(root: Path, request_id: str, entrypoint: str, body: dict[str, Any] | None) -> None:
    bundle = root / request_id
    bundle.mkdir(parents=True, exist_ok=True)
    request: dict[str, Any] = {
        "schema_version": 1,
        "entrypoint": entrypoint,
        "request_id": request_id,
        "model": MODEL,
        "backend": "actual",
        "actual_model_smoke": True,
        "sanitized": True,
    }
    if body is not None:
        request["http"] = {"method": "POST", "path": "/v1/chat/completions", "body": body}
    write_json(bundle / "request.json", request)
    write_json(bundle / "backend_selection.json", {"request_id": request_id, "backend": "actual", "model": MODEL})
    write_json(bundle / "bad_output_scan.json", {"request_id": request_id, "bad_output": False, "bad_text_count": 0, "reasons": []})


def chat_response(response_id: str, content: str) -> dict[str, Any]:
    return {
        "id": response_id,
        "model": MODEL,
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content, "reasoning": None}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def write_fixture_tree(root: Path) -> None:
    files = [
        {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name != "artifact_tree.json"
    ]
    write_json(root / "artifact_tree.json", self_hash({"schema_version": 1, "artifact_root": str(root), "file_count": len(files), "files": files}))


def create_fixture(root: Path) -> None:
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    shutil.copyfile(RUNNER_PATH, inputs / "run_scenarios.py")
    shutil.copyfile(HELPER_PATH, inputs / "openai_concurrency_quality_regression.py")
    shutil.copyfile(SCENARIO_MANIFEST_PATH, inputs / "scenario_manifest.json")
    binary_path = "/workspace/ferrum/target/release/ferrum"
    binary_sha = "3" * 64
    run_root = root / RUN_NAME
    run_root.mkdir()
    input_text = "\n".join(RUN_PROMPTS) + "\n/bye\n"
    (run_root / "input.txt").write_text(input_text, encoding="utf-8")
    run_argv = [binary_path, "run", "--backend", "cuda", "--temperature", "0.0", "--output-format", "jsonl", "--effective-config-json", str(run_root / "effective_config.json"), "--decision-trace-jsonl", str(run_root / "decision_trace.jsonl"), "--profile-jsonl", str(run_root / "observability/profile.jsonl"), "--profile-detail", "basic", "--memory-profile-jsonl", str(run_root / "observability/memory_profile.jsonl"), "--scheduler-trace-jsonl", str(run_root / "observability/scheduler_trace.jsonl"), "--request-dump-dir", str(run_root / "observability/request_dump"), "--profile-sample-rate", "1.0", "--disable-thinking", MODEL]
    write_json(run_root / "command.json", {"argv": run_argv, "binary_sha256": binary_sha, "child_env": {"HF_HOME": "/workspace/hf-cache", "NO_COLOR": "1"}, "cwd": "/workspace/ferrum", "env_policy": "remove_FERRUM_prefix", "removed_hidden_env_names": [], "stdin_path": str(run_root / "input.txt"), "stdin_sha256": hashlib.sha256(input_text.encode()).hexdigest()})
    run_ids = [f"{index + 1:08d}-1111-1111-1111-{index + 1:012d}" for index in range(len(RUN_PROMPTS))]
    events = [
        {"schema_version": 2, "event": "ready", "session_id": "session-fixture", "backend": "CUDA(0)"}
    ]
    for turn, (request_id, prompt, answer) in enumerate(
        zip(run_ids, RUN_PROMPTS, RUN_ANSWERS, strict=True)
    ):
        prompt_tokens = 10 * (turn + 1)
        completion_tokens = 1 if turn < len(RUN_ANSWERS) - 1 else 2
        events.extend(
            [
                {"schema_version": 2, "event": "user", "session_id": "session-fixture", "request_id": request_id, "turn": turn, "content": prompt},
                {"schema_version": 2, "event": "assistant", "session_id": "session-fixture", "request_id": request_id, "turn": turn, "content": answer, "reasoning": None, "finish_reason": "stop", "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}},
            ]
        )
    events.append(
        {"schema_version": 2, "event": "exit", "session_id": "session-fixture", "reason": "user_exit"}
    )
    (run_root / "stdout.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    (run_root / "stderr.log").write_text("clean run\n", encoding="utf-8")
    effective = {"schema_version": 1, "backend": "cuda", "cuda_device_count": 1, "selected_gpu_devices": [0], "selected_max_sequences": 1, "selected_admission_limit": 1, "model_capabilities": {"architecture": "qwen3_5"}, "hardware_capabilities": {"backend": "cuda", "compiled_features": {"cuda": True}}}
    write_json(run_root / "effective_config.json", effective)
    (run_root / "decision_trace.jsonl").write_text('{"event":"run_started"}\n', encoding="utf-8")
    run_obs = run_root / "observability"
    (run_obs / "request_dump").mkdir(parents=True)
    run_trace: list[dict[str, Any]] = []
    for request_id in run_ids:
        write_bundle(run_obs / "request_dump", request_id, "run", None)
        run_trace.extend(lifecycle_rows(request_id, "run"))
    (run_obs / "scheduler_trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in run_trace), encoding="utf-8")
    (run_obs / "profile.jsonl").write_text('{"event":"profile"}\n', encoding="utf-8")
    (run_obs / "memory_profile.jsonl").write_text('{"event":"memory"}\n', encoding="utf-8")

    serve_root = root / SERVE_NAME
    serve_root.mkdir()
    first_prompt = "Remember the codeword banana. Reply exactly OK."
    second_prompt = "What codeword did I ask you to remember? Answer with only the codeword."
    first_request = {"model": MODEL, "messages": [{"role": "user", "content": first_prompt}], "max_tokens": 128, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    second_request = {"model": MODEL, "messages": [{"role": "user", "content": first_prompt}, {"role": "assistant", "content": "OK"}, {"role": "user", "content": second_prompt}], "max_tokens": 128, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    write_json(serve_root / "turn1.request.json", first_request)
    write_json(serve_root / "turn2.request.json", second_request)
    write_json(serve_root / "turn1.json", chat_response("chatcmpl-turn1", "OK"))
    write_json(serve_root / "turn2.json", chat_response("chatcmpl-turn2", SECRET))

    concurrency_root = root / CONCURRENCY_NAME
    concurrency_root.mkdir()
    cell_summaries = []
    for concurrency in EXPECTED_CONCURRENCY:
        rows = []
        for index in range(concurrency):
            marker = f"ferrum{concurrency:02d}{index:02d}"
            checksum = f"S{(index + 1) ** 2:04d}"
            request_path = concurrency_root / f"c{concurrency}.quality.{index:03d}.request.json"
            response_path = concurrency_root / f"c{concurrency}.quality.{index:03d}.response.txt"
            write_json(request_path, expected_concurrency_request(concurrency, index))
            response = {
                "id": f"chatcmpl-c{concurrency}-{index}",
                "model": MODEL,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning": None,
                            "tool_calls": [
                                {
                                    "id": f"call-c{concurrency}-{index}",
                                    "type": "function",
                                    "function": {
                                        "name": "capture_quality_marker",
                                        "arguments": json.dumps(
                                            {"marker": marker, "checksum": checksum},
                                            separators=(",", ":"),
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
            }
            response_raw = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
            response_path.write_text(response_raw, encoding="utf-8")
            started = concurrency * 1_000_000_000 + index * 1_000_000
            finished = started + 100_000_000
            rows.append({"i": index, "status": 200, "json_ok": True, "marker": marker, "square": checksum, "tool_name": "capture_quality_marker", "parsed_marker": marker, "parsed_checksum": checksum, "marker_ok": True, "square_ok": True, "format_ok": True, "finish_reason": "tool_calls", "forbidden_text": None, "content_head": response_raw[:500], "request_artifact": str(request_path), "request_sha256": file_sha256(request_path), "response_artifact": str(response_path), "response_sha256": file_sha256(response_path), "response_size": response_path.stat().st_size, "started_monotonic_ns": started, "finished_monotonic_ns": finished, "duration_ms": 100.0})
        overlap_pair_count = concurrency * (concurrency - 1) // 2
        cell = {"concurrency": concurrency, "worker_count": min(concurrency, WORKER_LIMIT), "worker_limit": WORKER_LIMIT, "synchronized_start": concurrency > 1, "overlap_pair_count": overlap_pair_count, "max_in_flight": concurrency, "requests": concurrency, "status_200": concurrency, "json_ok": concurrency, "marker_ok": concurrency, "square_ok": concurrency, "format_ok": concurrency, "crosstalk": 0, "length_finishes": 0, "forbidden_count": 0, "passed": True, "rows": rows}
        write_json(concurrency_root / f"c{concurrency}.quality.json", cell)
        cell_summaries.append({key: value for key, value in cell.items() if key != "rows"})
    write_json(concurrency_root / "concurrency_quality_regression.json", {"model": MODEL, "cells": cell_summaries, "status": "pass"})

    scenario_rows = [
        {"name": RUN_NAME, "type": "run_multiturn", "status": "pass", "assistant_turns": len(RUN_PROMPTS), "length_finishes": 0, "used_default_max_tokens": True, "artifact": str(run_root / "result.json"), "duration_sec": 1.0},
        {"name": SERVE_NAME, "type": "serve_multiturn_recall", "status": "pass", "assistant_turns": 2, "recalled_secret": True, "artifact": str(serve_root / "result.json"), "duration_sec": 1.0},
        {"name": CONCURRENCY_NAME, "type": "serve_concurrency_quality", "status": "pass", "cells": cell_summaries, "artifact": str(concurrency_root / "result.json"), "duration_sec": 1.0},
    ]
    for row in scenario_rows:
        write_json(Path(row["artifact"]), row)

    server_obs = root / "observability/serve"
    (server_obs / "request_dump/serve-startup-fixture").mkdir(parents=True)
    write_bundle(server_obs / "request_dump", "serve-startup-fixture", "serve", None)
    serve_trace: list[dict[str, Any]] = []
    serve_bodies = [first_request, second_request]
    for concurrency in EXPECTED_CONCURRENCY:
        for index in range(concurrency):
            serve_bodies.append(expected_concurrency_request(concurrency, index))
    for index, body in enumerate(serve_bodies):
        request_id = f"33333333-3333-3333-3333-{index:012d}"
        write_bundle(
            server_obs / "request_dump",
            request_id,
            "serve",
            redacted_request_body(body),
        )
        serve_trace.extend(lifecycle_rows(request_id, "serve"))
    (server_obs / "scheduler_trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in serve_trace), encoding="utf-8")
    (server_obs / "profile.jsonl").write_text('{"event":"profile"}\n', encoding="utf-8")
    (server_obs / "memory_profile.jsonl").write_text('{"event":"memory"}\n', encoding="utf-8")
    observability = {"enabled": True, "roots": {"run": [str(run_obs)], "serve": str(server_obs)}, "profile_paths": [str(run_obs / "profile.jsonl"), str(run_obs / "memory_profile.jsonl"), str(run_obs / "scheduler_trace.jsonl"), str(server_obs / "profile.jsonl"), str(server_obs / "memory_profile.jsonl"), str(server_obs / "scheduler_trace.jsonl")], "scheduler_trace_paths": [str(run_obs / "scheduler_trace.jsonl"), str(server_obs / "scheduler_trace.jsonl")], "request_dump_dirs": [str(run_obs / "request_dump"), str(server_obs / "request_dump")]}
    write_json(root / "observability_summary.json", observability)
    write_json(root / "response_format_matrix_contract.json", {"schema_version": 1, "case_counts": {"json_schema": 0, "json_object": 0}, "unique_json_schema_count": 0})
    write_json(root / "server.effective_config.json", effective)
    (root / "server.decision_trace.jsonl").write_text('{"event":"server_started"}\n', encoding="utf-8")
    (root / "server.log").write_text("clean server\n", encoding="utf-8")
    write_json(root / "server.health.json", {"status": "pass", "http_status": 200})
    write_json(root / "server.health.after.json", {"status": "pass", "http_status": 200})
    input_artifacts = {
        "runner": {"path": str(inputs / "run_scenarios.py"), "sha256": file_sha256(inputs / "run_scenarios.py")},
        "manifest": {"path": str(inputs / "scenario_manifest.json"), "sha256": file_sha256(inputs / "scenario_manifest.json")},
        "concurrency_quality_helper": {"path": str(inputs / "openai_concurrency_quality_regression.py"), "sha256": file_sha256(inputs / "openai_concurrency_quality_regression.py")},
    }
    evidence = {key: {"path": str(root / filename), "size": (root / filename).stat().st_size, "sha256": file_sha256(root / filename)} for key, filename in {"effective_config": "server.effective_config.json", "decision_trace": "server.decision_trace.jsonl", "server_log": "server.log", "health_before": "server.health.json", "health_after": "server.health.after.json"}.items()}
    receipt = self_hash({"schema_version": 1, "mode": "start", "runner_argv": ["/usr/bin/python3", "/workspace/ferrum/scripts/release/run_scenarios.py", "--manifest", "/workspace/ferrum/scripts/release/scenarios/runtime_vnext_s2_multiturn_concurrency_cuda.json", "--out", str(root)], "runner_path": "/workspace/ferrum/scripts/release/run_scenarios.py", "runner_sha256": file_sha256(inputs / "run_scenarios.py"), "manifest_path": "/workspace/ferrum/scripts/release/scenarios/runtime_vnext_s2_multiturn_concurrency_cuda.json", "manifest_sha256": file_sha256(inputs / "scenario_manifest.json"), "cwd": "/workspace/ferrum", "git_sha": "a" * 40, "dirty_status": {"is_dirty": False, "status_short": []}, "input_artifacts": input_artifacts, "backend": "cuda", "model": MODEL, "selected_scenarios": [name for name, _ in SCENARIOS], "scenario_execution_phases": [{"phase": "run", "scenarios": [RUN_NAME], "started_at": "2026-08-02T00:00:00+00:00", "finished_at": "2026-08-02T00:00:10+00:00"}, {"phase": "serve", "scenarios": [SERVE_NAME, CONCURRENCY_NAME], "started_at": "2026-08-02T00:00:10+00:00", "finished_at": "2026-08-02T00:01:00+00:00"}], "scenario_count": 3, "failed": 0, "skipped": 0, "server_argv": [binary_path, "serve", "--host", "127.0.0.1", "--effective-config-json", str(root / "server.effective_config.json"), "--decision-trace-jsonl", str(root / "server.decision_trace.jsonl"), "--backend", "cuda", "--max-num-seqs", "1", MODEL], "binary_path": binary_path, "binary_sha256": binary_sha, "hardware": {"argv": ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader,nounits"], "returncode": 0, "stdout": "0, NVIDIA GeForce RTX 4090, GPU-fixture, 24564, 570.00\n", "stderr": ""}, "removed_hidden_env_names": [], "child_env": {"HF_HOME": "/workspace/hf-cache", "NO_COLOR": "1"}, "server_started_at": "2026-08-02T00:00:11+00:00", "server_finished_at": "2026-08-02T00:01:00+00:00", "server_returncode": -15, "evidence_files": evidence})
    write_json(root / "execution_receipt.json", receipt)
    summary = {"schema_version": 1, "status": "pass", "manifest": "/workspace/ferrum/scripts/release/scenarios/runtime_vnext_s2_multiturn_concurrency_cuda.json", "artifact_dir": str(root), "model": MODEL, "backend": "cuda", "base_url": "http://127.0.0.1:8000", "git_sha": "a" * 40, "dirty_status": {"is_dirty": False, "status_short": []}, "started_at": "2026-08-02T00:00:00+00:00", "finished_at": "2026-08-02T00:01:00+00:00", "scenario_count": 3, "manifest_scenario_count": 3, "requested_scenarios": [], "selected_scenarios": [name for name, _ in SCENARIOS], "failed": 0, "skipped": 0, "scenarios": scenario_rows, "response_format_matrix_contract": {"artifact": str(root / "response_format_matrix_contract.json"), "case_counts": {"json_schema": 0, "json_object": 0}, "unique_json_schema_count": 0}, "observability": observability, "execution_receipt": {"artifact": str(root / "execution_receipt.json"), "artifact_sha256": file_sha256(root / "execution_receipt.json"), "canonical_sha256": receipt["canonical_sha256"], "mode": "start", "runner_sha256": receipt["runner_sha256"], "manifest_sha256": receipt["manifest_sha256"], "binary_sha256": binary_sha}, "pass_line": f"BACKEND REGRESSION SMOKE PASS: {root}"}
    write_json(root / "summary.json", summary)
    write_fixture_tree(root)


def rewrite_tree(root: Path) -> None:
    (root / "artifact_tree.json").unlink(missing_ok=True)
    write_fixture_tree(root)


def rewrite_execution_receipt(
    root: Path, mutation: Callable[[dict[str, Any]], None]
) -> None:
    receipt_path = root / "execution_receipt.json"
    receipt = read_json(receipt_path)
    receipt.pop("canonical_sha256", None)
    receipt.pop("canonical_sha256_scope", None)
    mutation(receipt)
    receipt = self_hash(receipt)
    write_json(receipt_path, receipt)
    summary_path = root / "summary.json"
    summary = read_json(summary_path)
    summary_receipt = summary["execution_receipt"]
    summary_receipt["artifact_sha256"] = file_sha256(receipt_path)
    summary_receipt["canonical_sha256"] = receipt["canonical_sha256"]
    write_json(summary_path, summary)


def expect_reject(root: Path, mutation: Callable[[Path], None], expected: str) -> None:
    candidate = root.parent / f"mutation-{expected.replace(' ', '-')}"
    shutil.copytree(root, candidate)
    mutation(candidate)
    rewrite_tree(candidate)
    try:
        validate_source(candidate, "a" * 40)
    except ValidationError:
        return
    raise ValidationError(f"hostile mutation unexpectedly passed: {expected}")


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ferrum-s2-multiturn-concurrency-") as tmp:
        root = Path(tmp) / "fixture"
        root.mkdir()
        create_fixture(root)
        evidence = validate_source(root, "a" * 40)
        require(
            evidence["observability"]["execution_lifecycles"]
            == EXPECTED_RUN_REQUESTS + EXPECTED_SERVE_REQUESTS,
            "baseline lifecycle count mismatch",
        )
        source_tree_sha = file_sha256(root / "artifact_tree.json")
        nested_output = root / "validator-output"
        for invalid_output in (root, nested_output):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = run_checkpoint(root, invalid_output, "a" * 40)
            require(rc == 1, "unsafe checkpoint output layout unexpectedly passed")
            require(
                "checkpoint output must" in stderr.getvalue(),
                "unsafe checkpoint output rejection missing diagnostic",
            )
            require(
                file_sha256(root / "artifact_tree.json") == source_tree_sha,
                "unsafe checkpoint output mutated source artifact tree",
            )
        require(not nested_output.exists(), "unsafe nested checkpoint output was created")
        try:
            validate_source(root, "b" * 40)
        except ValidationError:
            pass
        else:
            raise ValidationError("stale SHA mutation unexpectedly passed")

        def bad_recall(candidate: Path) -> None:
            response = read_json(candidate / SERVE_NAME / "turn2.json")
            response["choices"][0]["message"]["content"] = "wrong"
            write_json(candidate / SERVE_NAME / "turn2.json", response)

        def bad_first_turn(candidate: Path) -> None:
            path = candidate / RUN_NAME / "stdout.jsonl"
            rows = read_jsonl(path)
            for row in rows:
                if row.get("event") == "assistant" and row.get("turn") == 0:
                    row["content"] = "not OK"
                    break
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

        def thinking_wrapper(candidate: Path) -> None:
            response = read_json(candidate / SERVE_NAME / "turn2.json")
            response["choices"][0]["message"]["content"] = (
                f"<think>hidden</think>{SECRET}"
            )
            write_json(candidate / SERVE_NAME / "turn2.json", response)

        def crosstalk(candidate: Path) -> None:
            cell = read_json(candidate / CONCURRENCY_NAME / "c4.quality.json")
            cell["crosstalk"] = 1
            write_json(candidate / CONCURRENCY_NAME / "c4.quality.json", cell)

        def legacy_trace(candidate: Path) -> None:
            path = candidate / "observability/serve/scheduler_trace.jsonl"
            rows = read_jsonl(path)
            for row in rows:
                if row.get("phase") == "vnext.request_completed":
                    row["attributes"]["execution_trace_source"] = "legacy"
                    break
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

        def plan_identity_drift(candidate: Path) -> None:
            path = candidate / "observability/serve/scheduler_trace.jsonl"
            rows = read_jsonl(path)
            for row in rows:
                attributes = row.get("attributes")
                if (
                    row.get("phase") == "vnext.operation_submitted"
                    and isinstance(attributes, dict)
                ):
                    attributes["plan_hash"] = "3" * 64
                    break
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

        def runtime_identity_drift(candidate: Path) -> None:
            path = candidate / "observability/serve/scheduler_trace.jsonl"
            rows = read_jsonl(path)
            for row in rows:
                attributes = row.get("attributes")
                if (
                    row.get("phase") == "vnext.operation_submitted"
                    and isinstance(attributes, dict)
                ):
                    attributes["runtime_implementation_fingerprint"] = "4" * 64
                    break
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

        def helper_drift(candidate: Path) -> None:
            path = candidate / "inputs/openai_concurrency_quality_regression.py"
            path.write_text(read_text(path) + "# drift\n", encoding="utf-8")

        def inherited_run_env(candidate: Path) -> None:
            path = candidate / RUN_NAME / "command.json"
            command = read_json(path)
            command["env_policy"] = "inherit-all"
            write_json(path, command)

        def changed_run_cwd(candidate: Path) -> None:
            path = candidate / RUN_NAME / "command.json"
            command = read_json(path)
            command["cwd"] = "/workspace/other"
            write_json(path, command)

        def explicit_run_max_tokens(candidate: Path) -> None:
            path = candidate / RUN_NAME / "command.json"
            command = read_json(path)
            argv = command["argv"]
            argv[argv.index("--temperature"):argv.index("--temperature")] = [
                "--max-tokens",
                "128",
            ]
            write_json(path, command)

        def swapped_run_request_ids(candidate: Path) -> None:
            path = candidate / RUN_NAME / "stdout.jsonl"
            rows = read_jsonl(path)
            assistants = [row for row in rows if row.get("event") == "assistant"]
            assistants[0]["request_id"], assistants[1]["request_id"] = (
                assistants[1]["request_id"],
                assistants[0]["request_id"],
            )
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

        def raw_reasoning_leak(candidate: Path) -> None:
            response_path = (
                candidate
                / CONCURRENCY_NAME
                / "c4.quality.000.response.txt"
            )
            response = json.loads(read_text(response_path))
            response["choices"][0]["message"]["reasoning_content"] = "leaked"
            raw = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
            response_path.write_text(raw, encoding="utf-8")
            cell_path = candidate / CONCURRENCY_NAME / "c4.quality.json"
            cell = read_json(cell_path)
            item = cell["rows"][0]
            item["response_sha256"] = file_sha256(response_path)
            item["response_size"] = response_path.stat().st_size
            item["content_head"] = raw[:500]
            write_json(cell_path, cell)

        def request_dump_drift(candidate: Path) -> None:
            dump_root = candidate / "observability/serve/request_dump"
            for bundle in sorted(path for path in dump_root.iterdir() if path.is_dir()):
                request_path = bundle / "request.json"
                request = read_json(request_path)
                tools = request.get("http", {}).get("body", {}).get("tools")
                if isinstance(tools, list) and tools:
                    tools[0]["function"]["name"] = "wrong_tool"
                    write_json(request_path, request)
                    return
            raise ValidationError("fixture has no concurrency request dump")

        def serial_client_intervals(candidate: Path) -> None:
            cell_path = candidate / CONCURRENCY_NAME / "c4.quality.json"
            cell = read_json(cell_path)
            for index, item in enumerate(cell["rows"]):
                started = 10_000_000_000 + index * 200_000_000
                item["started_monotonic_ns"] = started
                item["finished_monotonic_ns"] = started + 100_000_000
                item["duration_ms"] = 100.0
            cell["overlap_pair_count"] = 0
            cell["max_in_flight"] = 1
            write_json(cell_path, cell)
            summary_cell = {key: value for key, value in cell.items() if key != "rows"}
            aggregate_path = candidate / CONCURRENCY_NAME / "concurrency_quality_regression.json"
            aggregate = read_json(aggregate_path)
            aggregate["cells"][1] = summary_cell
            write_json(aggregate_path, aggregate)
            result_path = candidate / CONCURRENCY_NAME / "result.json"
            result = read_json(result_path)
            result["cells"][1] = summary_cell
            write_json(result_path, result)
            summary_path = candidate / "summary.json"
            summary = read_json(summary_path)
            summary["scenarios"][2]["cells"][1] = summary_cell
            write_json(summary_path, summary)

        def server_overlaps_run_phase(candidate: Path) -> None:
            rewrite_execution_receipt(
                candidate,
                lambda receipt: receipt.__setitem__(
                    "server_started_at", "2026-08-02T00:00:05+00:00"
                ),
            )

        expect_reject(root, bad_recall, "bad recall")
        expect_reject(root, bad_first_turn, "bad first turn")
        expect_reject(root, thinking_wrapper, "thinking wrapper")
        expect_reject(root, crosstalk, "crosstalk")
        expect_reject(root, legacy_trace, "legacy trace")
        expect_reject(root, plan_identity_drift, "plan identity drift")
        expect_reject(root, runtime_identity_drift, "runtime identity drift")
        expect_reject(root, helper_drift, "helper drift")
        expect_reject(root, inherited_run_env, "inherited run env")
        expect_reject(root, changed_run_cwd, "changed run cwd")
        expect_reject(root, explicit_run_max_tokens, "explicit run max tokens")
        expect_reject(root, swapped_run_request_ids, "swapped run ids")
        expect_reject(root, raw_reasoning_leak, "raw reasoning leak")
        expect_reject(root, request_dump_drift, "request dump drift")
        expect_reject(root, serial_client_intervals, "serial client intervals")
        expect_reject(root, server_overlaps_run_phase, "server overlaps run phase")
    print(SELFTEST_PASS_LINE)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-git-sha")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    if args.source is None or args.out is None:
        parser.error("--source and --out are required unless --self-test is used")
    if args.expected_git_sha is not None and GIT_SHA_RE.fullmatch(args.expected_git_sha) is None:
        parser.error("--expected-git-sha must be 40 lowercase hex characters")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        try:
            self_test()
            return 0
        except (OSError, ValidationError) as error:
            print(f"{SELFTEST_PASS_LINE.replace(' PASS', ' FAIL')}: {error}", file=sys.stderr)
            return 1
    return run_checkpoint(args.source, args.out, args.expected_git_sha)


if __name__ == "__main__":
    raise SystemExit(main())
