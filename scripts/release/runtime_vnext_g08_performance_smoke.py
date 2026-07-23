#!/usr/bin/env python3
"""Validate the shared low-cost G08 model-migration performance smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = 1
CONTRACT_SCHEMA = "ferrum.runtime-vnext.g08.performance-smoke-diagnostic.v1"
PASS_PREFIX = "FERRUM RUNTIME VNEXT G08 PERFORMANCE SMOKE PASS"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
MODEL_KEYS = {
    "m1-qwen35-4b",
    "m2-qwen35-35b-a3b",
    "m3-qwen3-30b-a3b",
}
BACKEND_CONCURRENCY = {
    "cuda": (1, 32),
    "metal": (1, 16),
}
BASELINE_THRESHOLDS = {
    "legacy": 0.90,
    "external": 0.70,
}
QUALITY_KEYS = {
    "bad_output",
    "malformed_stream",
    "missing_done",
    "duplicate_done",
    "zero_output_tokens",
    "stream_bulk_flush",
    "http_500",
    "panic",
}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid JSON {path}: {error}") from error
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def required_file(root: Path, relative: str, *, nonempty: bool = True) -> Path:
    path = root / relative
    require(path.is_file(), f"missing artifact: {relative}")
    if nonempty:
        require(path.stat().st_size > 0, f"empty artifact: {relative}")
    return path


def file_record(root: Path, relative: str, *, nonempty: bool = True) -> dict[str, Any]:
    path = required_file(root, relative, nonempty=nonempty)
    return {
        "path": relative,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def command(path: Path) -> list[str]:
    try:
        parts = shlex.split(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValidationError(f"invalid command receipt {path}: {error}") from error
    require(parts, f"empty command receipt: {path}")
    return parts


def flag_value(parts: list[str], flag: str) -> str:
    indexes = [index for index, part in enumerate(parts) if part == flag]
    require(len(indexes) == 1, f"command must contain {flag} exactly once")
    index = indexes[0]
    require(index + 1 < len(parts), f"command {flag} has no value")
    return parts[index + 1]


def require_flag(parts: list[str], flag: str) -> None:
    require(parts.count(flag) == 1, f"command must contain {flag} exactly once")


def require_zero_quality(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == QUALITY_KEYS, f"{label} quality keys drifted")
    for key, count in value.items():
        require(count == 0, f"{label}.{key} must be 0")


def validate_report(
    report: dict[str, Any],
    *,
    implementation: str,
    concurrency: int,
) -> tuple[float, dict[str, Any]]:
    label = f"{implementation}/c{concurrency}"
    require(report.get("backend") in {"cuda", "metal"}, f"{label} backend is invalid")
    require(report.get("scenario") == "closed_loop", f"{label} is not closed_loop")
    require(report.get("concurrency") == concurrency, f"{label} concurrency mismatch")
    require(report.get("n_prompt") == 64, f"{label} n_prompt must be 64")
    require(report.get("n_gen") == 32, f"{label} n_gen must be 32")
    require(report.get("n_repeats") == 3, f"{label} n_repeats must be 3")
    require(report.get("n_requests_per_run") == 100, f"{label} requests must be 100")
    require(report.get("warmup_requests") == 10, f"{label} warmup must be 10")
    require(
        report.get("output_token_count_source") == "usage",
        f"{label} output token source must be usage",
    )
    require(
        report.get("completed_per_run") == [100, 100, 100],
        f"{label} completed_per_run must be 100/100/100",
    )
    for key in (
        "errored_per_run",
        "bad_output_per_run",
        "malformed_stream_per_run",
        "missing_done_per_run",
        "duplicate_done_per_run",
        "zero_output_tokens_per_run",
        "stream_bulk_flush_per_run",
        "http_500_per_run",
        "panic_per_run",
    ):
        require(report.get(key) == [0, 0, 0], f"{label} {key} must be 0/0/0")
    quality_rows = report.get("quality_issues_per_run")
    require(
        isinstance(quality_rows, list) and len(quality_rows) == 3,
        f"{label} quality_issues_per_run must contain three rows",
    )
    for index, row in enumerate(quality_rows, start=1):
        require_zero_quality(row, f"{label}/repeat{index}.quality")

    actual = report.get("actual_input_tokens")
    require(isinstance(actual, dict), f"{label} actual_input_tokens must be an object")
    for key in ("requested", "min", "max"):
        require(actual.get(key) == 64, f"{label} actual_input_tokens.{key} must be 64")

    input_rows = report.get("actual_input_tokens_per_request")
    output_rows = report.get("output_tokens_per_request")
    require(
        isinstance(input_rows, list) and len(input_rows) == 3,
        f"{label} input token rows must contain three repeats",
    )
    require(
        isinstance(output_rows, list) and len(output_rows) == 3,
        f"{label} output token rows must contain three repeats",
    )
    for index, values in enumerate(input_rows, start=1):
        require(
            isinstance(values, list) and values == [64] * 100,
            f"{label}/repeat{index} input lengths are not exactly 64",
        )
    for index, values in enumerate(output_rows, start=1):
        require(
            isinstance(values, list) and values == [32] * 100,
            f"{label}/repeat{index} output lengths are not exactly 32",
        )

    repeats = report.get("repeat_metrics")
    require(
        isinstance(repeats, list) and len(repeats) == 3,
        f"{label} repeat_metrics must contain three rows",
    )
    throughputs: list[float] = []
    for index, row in enumerate(repeats, start=1):
        require(isinstance(row, dict), f"{label}/repeat{index} must be an object")
        require(row.get("repeat") == index, f"{label}/repeat{index} ordinal mismatch")
        require(row.get("expected_requests") == 100, f"{label}/repeat{index} expected requests")
        require(row.get("completed_requests") == 100, f"{label}/repeat{index} completion")
        require(row.get("errored_requests") == 0, f"{label}/repeat{index} errors")
        require(row.get("warmup_expected") == 10, f"{label}/repeat{index} warmup expected")
        require(row.get("warmup_completed") == 10, f"{label}/repeat{index} warmup completion")
        require(row.get("warmup_errored") == 0, f"{label}/repeat{index} warmup errors")
        require(row.get("actual_input_tokens") == 6400, f"{label}/repeat{index} input total")
        require(row.get("output_tokens") == 3200, f"{label}/repeat{index} output total")
        require(
            row.get("output_token_count_source") == "usage",
            f"{label}/repeat{index} output token source",
        )
        require_zero_quality(row.get("quality_issues"), f"{label}/repeat{index}.quality")
        require_zero_quality(
            row.get("warmup_quality_issues"),
            f"{label}/repeat{index}.warmup_quality",
        )
        throughput = row.get("output_throughput_tps")
        require(
            isinstance(throughput, (int, float))
            and not isinstance(throughput, bool)
            and math.isfinite(float(throughput))
            and float(throughput) > 0,
            f"{label}/repeat{index} output throughput is invalid",
        )
        throughputs.append(float(throughput))

    median = statistics.median(throughputs)
    return median, {
        "implementation": implementation,
        "concurrency": concurrency,
        "repeat_output_throughput_tps": throughputs,
        "median_output_throughput_tps": median,
    }


def validate_bench_command(
    parts: list[str],
    *,
    contract: dict[str, Any],
    implementation: str,
    concurrency: int,
) -> None:
    require(len(parts) >= 2 and parts[1] == "bench-serve", "benchmark must use ferrum bench-serve")
    expected = {
        "--model": contract["request_model"],
        "--target-backend": contract["backend"],
        "--http-connection-mode": "fresh",
        "--concurrency": str(concurrency),
        "--dataset": "random",
        "--random-input-len": "64",
        "--random-output-len": "32",
        "--num-prompts": "100",
        "--warmup-requests": "10",
        "--n-repeats": "3",
        "--seed": "9271",
        "--timeout": "600",
    }
    for flag, value in expected.items():
        require(
            flag_value(parts, flag) == value,
            f"{implementation}/c{concurrency} {flag} differs from G08 contract",
        )
    for flag in ("--fail-on-error", "--ignore-eos"):
        require_flag(parts, flag)
    require("--require-ci" not in parts, "G08 smoke must not use --require-ci")
    out_value = flag_value(parts, "--out")
    require(
        out_value.endswith(f"/{implementation}/bench-c{concurrency}.json"),
        f"{implementation}/c{concurrency} --out is not artifact-local",
    )


def validate_server_commands(
    baseline: list[str],
    candidate: list[str],
    contract: dict[str, Any],
) -> None:
    for label, parts in (("baseline", baseline), ("candidate", candidate)):
        require("serve" in parts, f"{label} command is not ferrum serve")
        serve_index = parts.index("serve")
        require(serve_index + 1 < len(parts), f"{label} serve model is missing")
        require(
            parts[serve_index + 1] == contract["request_model"],
            f"{label} serve model differs from request_model",
        )
        require(
            flag_value(parts, "--backend") == contract["backend"],
            f"{label} backend differs from contract",
        )
        require(
            flag_value(parts, "--max-num-seqs") == str(contract["typed_active_cap"]),
            f"{label} typed active cap differs from contract",
        )
    for flag in (
        "--host",
        "--port",
        "--backend",
        "--gpu-devices",
        "--gpu-memory-utilization",
        "--max-num-seqs",
    ):
        require(
            flag_value(baseline, flag) == flag_value(candidate, flag),
            f"baseline/candidate server {flag} differs",
        )


def validate_contract(contract: dict[str, Any]) -> tuple[str, tuple[int, ...], float]:
    require(contract.get("schema") == CONTRACT_SCHEMA, "performance smoke schema mismatch")
    require(contract.get("model_key") in MODEL_KEYS, "unsupported G08 model_key")
    backend = contract.get("backend")
    require(backend in BACKEND_CONCURRENCY, "unsupported performance smoke backend")
    concurrency = BACKEND_CONCURRENCY[backend]
    require(
        contract.get("requested_concurrency") == list(concurrency),
        f"{backend} G08 concurrency cells must be {list(concurrency)}",
    )
    baseline_kind = contract.get("baseline_kind")
    require(baseline_kind in BASELINE_THRESHOLDS, "baseline_kind must be legacy or external")
    threshold = BASELINE_THRESHOLDS[baseline_kind]
    require(
        contract.get("threshold_candidate_over_legacy") == threshold,
        f"{baseline_kind} threshold must be {threshold:.2f}",
    )
    for key, expected in (
        ("random_input_tokens", 64),
        ("random_output_tokens", 32),
        ("requests_per_repeat", 100),
        ("warmup_requests", 10),
        ("repeats", 3),
        ("seed", 9271),
        ("diagnostic_only", True),
        ("performance_claim", False),
        ("candidate_source_dirty", False),
    ):
        require(contract.get(key) == expected, f"contract {key} must be {expected!r}")
    require(contract.get("order") == [baseline_kind, "candidate"], "execution order must be baseline then candidate")
    require(
        isinstance(contract.get("typed_active_cap"), int)
        and not isinstance(contract["typed_active_cap"], bool)
        and contract["typed_active_cap"] > 0,
        "typed_active_cap must be a positive integer",
    )
    require(
        isinstance(contract.get("request_model"), str) and contract["request_model"],
        "request_model is missing",
    )
    for key in ("source_git_sha", "source_tree_sha"):
        require(
            isinstance(contract.get(key), str) and GIT_SHA_RE.fullmatch(contract[key]),
            f"contract {key} is invalid",
        )
    for key in ("candidate_binary_sha256", "baseline_binary_sha256"):
        require(
            isinstance(contract.get(key), str) and SHA256_RE.fullmatch(contract[key]),
            f"contract {key} is invalid",
        )
    return baseline_kind, concurrency, threshold


def validate_artifact(root: Path) -> dict[str, Any]:
    root = root.resolve()
    require(root.is_dir(), f"artifact root does not exist: {root}")
    required_file(root, "done", nonempty=False)
    contract_path = required_file(root, "control/contract.json")
    contract = read_json(contract_path)
    baseline_kind, concurrency_cells, threshold = validate_contract(contract)

    require(
        required_file(root, "control/git-head.txt").read_text().strip()
        == contract["source_git_sha"],
        "git HEAD does not match contract",
    )
    require(
        required_file(root, "control/git-tree.txt").read_text().strip()
        == contract["source_tree_sha"],
        "git tree does not match contract",
    )
    require(
        required_file(root, "control/git-status.txt", nonempty=False).read_text().strip() == "",
        "candidate source was dirty",
    )
    binary_receipt = required_file(root, "control/binary-sha256.txt").read_text()
    for key in ("candidate_binary_sha256", "baseline_binary_sha256"):
        require(contract[key] in binary_receipt, f"{key} is absent from binary receipt")
    for relative in (
        "control/nvidia-smi.before.txt",
        "control/nvidia-smi.after.txt",
        "control/nvcc-version.txt",
        "control/sanitized-env.txt",
    ):
        required_file(root, relative)

    baseline_dir = root / baseline_kind
    candidate_dir = root / "candidate"
    require(baseline_dir.is_dir(), f"missing {baseline_kind} artifact directory")
    require(candidate_dir.is_dir(), "missing candidate artifact directory")
    baseline_server = command(required_file(root, f"{baseline_kind}/server.command.txt"))
    candidate_server = command(required_file(root, "candidate/server.command.txt"))
    validate_server_commands(baseline_server, candidate_server, contract)

    cells: list[dict[str, Any]] = []
    medians: dict[str, dict[int, float]] = {baseline_kind: {}, "candidate": {}}
    report_paths: dict[str, list[Path]] = {baseline_kind: [], "candidate": []}
    benchmark_client: str | None = None
    for implementation in (baseline_kind, "candidate"):
        for relative in (
            f"{implementation}/effective-config.json",
            f"{implementation}/health.ready.json",
            f"{implementation}/models.json",
            f"{implementation}/scheduler-trace.jsonl",
            f"{implementation}/server.log",
        ):
            required_file(root, relative)
        for concurrency in concurrency_cells:
            report_relative = f"{implementation}/bench-c{concurrency}.json"
            report_path = required_file(root, report_relative)
            report_paths[implementation].append(report_path)
            report = read_json(report_path)
            require(
                report.get("backend") == contract["backend"],
                f"{implementation}/c{concurrency} backend differs from contract",
            )
            median, cell = validate_report(
                report,
                implementation=implementation,
                concurrency=concurrency,
            )
            cell["report"] = file_record(root, report_relative)
            cells.append(cell)
            medians[implementation][concurrency] = median
            command_parts = command(
                required_file(root, f"{implementation}/bench-c{concurrency}.command.txt")
            )
            validate_bench_command(
                command_parts,
                contract=contract,
                implementation=implementation,
                concurrency=concurrency,
            )
            if benchmark_client is None:
                benchmark_client = command_parts[0]
            require(
                command_parts[0] == benchmark_client,
                "all cells must use the same benchmark client binary",
            )
            required_file(root, f"{implementation}/health.after-c{concurrency}.json")

    require(
        max(path.stat().st_mtime_ns for path in report_paths[baseline_kind])
        < min(path.stat().st_mtime_ns for path in report_paths["candidate"]),
        "artifact timestamps do not prove baseline-before-candidate order",
    )

    ratios = []
    for concurrency in concurrency_cells:
        ratio = medians["candidate"][concurrency] / medians[baseline_kind][concurrency]
        require(
            math.isfinite(ratio) and ratio >= threshold,
            f"c{concurrency} candidate/{baseline_kind}={ratio:.6f} is below {threshold:.2f}",
        )
        ratios.append(
            {
                "concurrency": concurrency,
                f"candidate_over_{baseline_kind}": ratio,
                "threshold": threshold,
                "passes": True,
            }
        )
    return {
        "contract": contract,
        "baseline_kind": baseline_kind,
        "benchmark_client": benchmark_client,
        "cells": cells,
        "ratios": ratios,
    }


def write_checkpoint(root: Path, out: Path) -> dict[str, Any]:
    summary = validate_artifact(root)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not (out / "manifest.json").exists(), "checkpoint output already contains manifest.json")
    root = root.resolve()
    contract = summary["contract"]
    pass_line = f"{PASS_PREFIX}: {out}"
    validation = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g08_performance_smoke_validation",
        "status": "pass",
        "validated_at": iso_now(),
        "source_git_sha": contract["source_git_sha"],
        "source_tree_sha": contract["source_tree_sha"],
        "model_key": contract["model_key"],
        "backend": contract["backend"],
        "candidate_binary_sha256": contract["candidate_binary_sha256"],
        "baseline_binary_sha256": contract["baseline_binary_sha256"],
        "artifact_root": str(root),
        "contract": file_record(root, "control/contract.json"),
        "summary": {key: value for key, value in summary.items() if key != "contract"},
        "does_not_prove": [
            "G09 formal ABBA-BAAB performance",
            "performance no-regression",
            "external competitiveness when baseline_kind is legacy",
            "G10 release readiness",
        ],
        "pass_line": pass_line,
    }
    validation_path = out / "validation.json"
    write_json(validation_path, validation)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "runtime_vnext_g08_performance_smoke_manifest",
        "lane": "runtime-vnext-g08-performance-smoke",
        "status": "pass",
        "canonical": True,
        "source_git_sha": contract["source_git_sha"],
        "source_tree_sha": contract["source_tree_sha"],
        "dirty": False,
        "model_key": contract["model_key"],
        "backend": contract["backend"],
        "artifact_dir": str(out),
        "validation": {
            "path": str(validation_path),
            "sha256": file_sha256(validation_path),
        },
        "summary": validation["summary"],
        "does_not_prove": validation["does_not_prove"],
        "pass_line": pass_line,
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def fixture_report(concurrency: int, throughput: float) -> dict[str, Any]:
    quality = {key: 0 for key in sorted(QUALITY_KEYS)}
    repeats = []
    for repeat in range(1, 4):
        repeats.append(
            {
                "repeat": repeat,
                "duration_s": 1.0,
                "expected_requests": 100,
                "completed_requests": 100,
                "errored_requests": 0,
                "warmup_expected": 10,
                "warmup_completed": 10,
                "warmup_errored": 0,
                "actual_input_tokens": 6400,
                "output_tokens": 3200,
                "output_token_count_source": "usage",
                "output_throughput_tps": throughput + repeat,
                "quality_issues": copy.deepcopy(quality),
                "warmup_quality_issues": copy.deepcopy(quality),
            }
        )
    return {
        "model": "fixture",
        "backend": "cuda",
        "scenario": "closed_loop",
        "concurrency": concurrency,
        "n_prompt": 64,
        "n_gen": 32,
        "n_repeats": 3,
        "n_requests_per_run": 100,
        "warmup_requests": 10,
        "actual_input_tokens": {"requested": 64, "min": 64, "max": 64, "mean": 64.0},
        "actual_input_tokens_per_request": [[64] * 100 for _ in range(3)],
        "output_tokens_per_request": [[32] * 100 for _ in range(3)],
        "output_token_count_source": "usage",
        "repeat_metrics": repeats,
        "completed_per_run": [100, 100, 100],
        "errored_per_run": [0, 0, 0],
        "bad_output_per_run": [0, 0, 0],
        "malformed_stream_per_run": [0, 0, 0],
        "missing_done_per_run": [0, 0, 0],
        "duplicate_done_per_run": [0, 0, 0],
        "zero_output_tokens_per_run": [0, 0, 0],
        "stream_bulk_flush_per_run": [0, 0, 0],
        "http_500_per_run": [0, 0, 0],
        "panic_per_run": [0, 0, 0],
        "quality_issues_per_run": [copy.deepcopy(quality) for _ in range(3)],
    }


def fixture_artifact(root: Path) -> None:
    source = "1" * 40
    tree = "2" * 40
    baseline_sha = "3" * 64
    candidate_sha = "4" * 64
    model = "/models/m2"
    contract = {
        "schema": CONTRACT_SCHEMA,
        "model_key": "m2-qwen35-35b-a3b",
        "backend": "cuda",
        "baseline_kind": "legacy",
        "source_git_sha": source,
        "source_tree_sha": tree,
        "candidate_source_dirty": False,
        "candidate_binary_sha256": candidate_sha,
        "baseline_binary_sha256": baseline_sha,
        "request_model": model,
        "requested_concurrency": [1, 32],
        "typed_active_cap": 16,
        "random_input_tokens": 64,
        "random_output_tokens": 32,
        "requests_per_repeat": 100,
        "warmup_requests": 10,
        "repeats": 3,
        "seed": 9271,
        "threshold_candidate_over_legacy": 0.90,
        "diagnostic_only": True,
        "performance_claim": False,
        "order": ["legacy", "candidate"],
    }
    write_json(root / "control/contract.json", contract)
    (root / "control/git-head.txt").write_text(source + "\n")
    (root / "control/git-tree.txt").write_text(tree + "\n")
    (root / "control/git-status.txt").write_text("")
    (root / "control/binary-sha256.txt").write_text(
        f"{baseline_sha}  /baseline/ferrum\n{candidate_sha}  /candidate/ferrum\n"
    )
    for name in (
        "nvidia-smi.before.txt",
        "nvidia-smi.after.txt",
        "nvcc-version.txt",
        "sanitized-env.txt",
    ):
        (root / "control" / name).write_text("fixture\n")
    (root / "done").write_text("")
    for ordinal, implementation in enumerate(("legacy", "candidate")):
        directory = root / implementation
        directory.mkdir(parents=True, exist_ok=True)
        binary = f"/{implementation}/ferrum"
        server = [
            binary,
            "serve",
            model,
            "--backend",
            "cuda",
            "--host",
            "127.0.0.1",
            "--port",
            "18080",
            "--gpu-devices",
            "0",
            "--gpu-memory-utilization",
            "0.90",
            "--max-num-seqs",
            "16",
        ]
        (directory / "server.command.txt").write_text(shlex.join(server) + "\n")
        for name in (
            "effective-config.json",
            "health.ready.json",
            "models.json",
        ):
            write_json(directory / name, {"fixture": True})
        (directory / "scheduler-trace.jsonl").write_text("{}\n")
        (directory / "server.log").write_text("fixture\n")
        for concurrency in (1, 32):
            report_path = directory / f"bench-c{concurrency}.json"
            throughput = (10.0 if concurrency == 1 else 100.0) * (
                1.0 if implementation == "legacy" else 0.95
            )
            write_json(report_path, fixture_report(concurrency, throughput))
            timestamp = 1000 + ordinal * 100 + concurrency
            os.utime(report_path, (timestamp, timestamp))
            bench = [
                "/candidate/ferrum",
                "bench-serve",
                "--base-url",
                "http://127.0.0.1:18080",
                "--model",
                model,
                "--tokenizer",
                "/tokenizer",
                "--target-backend",
                "cuda",
                "--http-connection-mode",
                "fresh",
                "--concurrency",
                str(concurrency),
                "--dataset",
                "random",
                "--random-input-len",
                "64",
                "--random-output-len",
                "32",
                "--num-prompts",
                "100",
                "--warmup-requests",
                "10",
                "--n-repeats",
                "3",
                "--seed",
                "9271",
                "--timeout",
                "600",
                "--fail-on-error",
                "--ignore-eos",
                "--out",
                f"/artifact/{implementation}/bench-c{concurrency}.json",
            ]
            (directory / f"bench-c{concurrency}.command.txt").write_text(
                shlex.join(bench) + "\n"
            )
            write_json(directory / f"health.after-c{concurrency}.json", {"fixture": True})


def expect_reject(
    name: str,
    mutate: Callable[[Path], None],
    marker: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"g08-perf-{name}-") as temp:
        root = Path(temp)
        fixture_artifact(root)
        mutate(root)
        try:
            validate_artifact(root)
        except ValidationError as error:
            require(marker.lower() in str(error).lower(), f"{name} rejected unexpectedly: {error}")
            return
        raise AssertionError(f"{name} unexpectedly passed")


def make_candidate_c32_slow(root: Path) -> None:
    path = root / "candidate/bench-c32.json"
    report = read_json(path)
    for row in report["repeat_metrics"]:
        row["output_throughput_tps"] = 50.0
    write_json(path, report)


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="g08-perf-valid-") as temp:
        root = Path(temp)
        fixture_artifact(root)
        summary = validate_artifact(root)
        require(len(summary["cells"]) == 4, "fixture must produce four cells")
        require(len(summary["ratios"]) == 2, "fixture must produce two ratios")
    expect_reject(
        "error",
        lambda root: write_json(
            root / "candidate/bench-c32.json",
            {
                **read_json(root / "candidate/bench-c32.json"),
                "errored_per_run": [0, 1, 0],
            },
        ),
        "errored_per_run",
    )
    expect_reject(
        "slow",
        make_candidate_c32_slow,
        "candidate/legacy",
    )
    expect_reject(
        "dirty",
        lambda root: (root / "control/git-status.txt").write_text(" M source.rs\n"),
        "dirty",
    )
    expect_reject(
        "require-ci",
        lambda root: (root / "candidate/bench-c32.command.txt").write_text(
            (root / "candidate/bench-c32.command.txt").read_text().strip()
            + " --require-ci\n"
        ),
        "must not use --require-ci",
    )
    print("FERRUM RUNTIME VNEXT G08 PERFORMANCE SMOKE SELFTEST PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.artifact_root is None or args.out is None:
        parser.error("--artifact-root and --out are required")
    try:
        manifest = write_checkpoint(args.artifact_root, args.out)
    except (ValidationError, OSError, ValueError) as error:
        print(f"{PASS_PREFIX.replace('PASS', 'FAIL')}: {args.out}: {error}", file=sys.stderr)
        return 1
    print(manifest["pass_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
