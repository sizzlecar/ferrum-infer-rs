#!/usr/bin/env python3
"""Collect and validate G08A real-Metal operation/state numerical evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import runtime_vnext_numerical_tolerances as tolerances


REPO_ROOT = Path(__file__).resolve().parents[2]
PASS_PREFIX = "FERRUM RUNTIME VNEXT G08A METAL OP NUMERICS PASS"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT G08A METAL OP NUMERICS FAIL"
SELFTEST_PASS = "FERRUM RUNTIME VNEXT G08A METAL OP NUMERICS SELFTEST PASS"
METRICS_PREFIX = "FERRUM VNEXT NUMERICAL METRICS: "
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
METRIC_FIELDS = frozenset(
    {
        "tolerance_id",
        "row_fingerprint",
        "element_count",
        "shape",
        "logical_dtype",
        "oracle_precision",
        "actual_f32_sha256",
        "expected_f32_sha256",
        "actual_nan_count",
        "actual_inf_count",
        "expected_nan_count",
        "expected_inf_count",
        "max_abs",
        "max_relative_error",
        "max_relative_error_denominator_floor",
        "relative_l2",
        "cosine",
    }
)


class EvidenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read JSON {path}: {error}") from error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    process = subprocess.run(
        ["git", "-c", "core.preloadindex=false", "-c", "index.threads=1", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(process.returncode == 0, process.stderr.strip() or "git command failed")
    return process.stdout.strip()


def exact_object(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == fields, f"{label} fields differ: {sorted(set(value) ^ fields)}")
    return value


def finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def operation_rows(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {
        row["tolerance_id"]: row
        for row in catalog["rows"]
        if str(row["basis"]["source_path"]).startswith("crates/")
    }
    require(len(rows) == 27, f"expected 27 Metal op/state tolerance rows, got {len(rows)}")
    require(
        all(row["basis"]["kind"] == "checked_in_conformance_test" for row in rows.values()),
        "operation evidence contains a non-conformance-test basis",
    )
    return rows


def validate_metric(value: Any, row: dict[str, Any], label: str) -> dict[str, Any]:
    metric = exact_object(value, METRIC_FIELDS, label)
    require(metric["tolerance_id"] == row["tolerance_id"], f"{label} tolerance id differs")
    require(
        metric["row_fingerprint"]
        == row["row_fingerprint"]
        == tolerances.row_fingerprint(row),
        f"{label} row fingerprint differs",
    )
    shape = metric["shape"]
    require(
        isinstance(shape, list)
        and bool(shape)
        and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in shape),
        f"{label} shape is invalid",
    )
    require(
        isinstance(metric["element_count"], int)
        and not isinstance(metric["element_count"], bool)
        and metric["element_count"] == math.prod(shape),
        f"{label} element count differs from shape",
    )
    require(metric["logical_dtype"] == row["dtype"], f"{label} dtype differs")
    require(metric["oracle_precision"] == row["oracle_precision"] == "fp32", f"{label} oracle differs")
    for field in ("actual_f32_sha256", "expected_f32_sha256"):
        require(
            isinstance(metric[field], str) and SHA256_RE.fullmatch(metric[field]) is not None,
            f"{label}.{field} is not SHA256",
        )
    invariants = row["invariants"]
    for field, limit_name in (
        ("actual_nan_count", "max_nan"),
        ("expected_nan_count", "max_nan"),
        ("actual_inf_count", "max_inf"),
        ("expected_inf_count", "max_inf"),
    ):
        count = metric[field]
        require(
            isinstance(count, int)
            and not isinstance(count, bool)
            and 0 <= count <= invariants[limit_name],
            f"{label}.{field} exceeds catalog invariant",
        )
    max_abs = finite(metric["max_abs"], f"{label}.max_abs")
    relative_l2 = finite(metric["relative_l2"], f"{label}.relative_l2")
    cosine = finite(metric["cosine"], f"{label}.cosine")
    finite(metric["max_relative_error"], f"{label}.max_relative_error")
    require(
        metric["max_relative_error_denominator_floor"] == 1.0e-12,
        f"{label} relative-error floor differs",
    )
    bounds = row["bounds"]
    require(max_abs <= bounds["max_abs_max"], f"{label}.max_abs exceeds catalog")
    require(relative_l2 <= bounds["relative_l2_max"], f"{label}.relative_l2 exceeds catalog")
    require(cosine >= bounds["cosine_min"], f"{label}.cosine is below catalog")
    return metric


def parse_log(log: str, catalog: dict[str, Any]) -> dict[str, Any]:
    rows = operation_rows(catalog)
    required_tests = sorted({row["basis"]["test_name"] for row in rows.values()})
    require("test result: ok." in log, "cargo test did not report an ok result")
    require("skipping" not in log.lower(), "real-Metal conformance log contains a skip")
    for test_name in required_tests:
        pattern = re.compile(rf"^test .*::{re.escape(test_name)} \.\.\. ok$", re.MULTILINE)
        require(pattern.search(log) is not None, f"required Metal test did not pass: {test_name}")

    observed: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(log.splitlines(), start=1):
        marker = line.find(METRICS_PREFIX)
        if marker < 0:
            continue
        payload = line[marker + len(METRICS_PREFIX) :]
        try:
            envelope = json.loads(payload)
        except json.JSONDecodeError as error:
            raise EvidenceError(f"line {line_number} contains malformed numerical JSON") from error
        envelope = exact_object(envelope, frozenset({"label", "metrics"}), f"line {line_number}")
        require(isinstance(envelope["label"], str) and envelope["label"], f"line {line_number} label is empty")
        raw_metric = envelope["metrics"]
        require(isinstance(raw_metric, dict), f"line {line_number} metrics must be an object")
        tolerance_id = raw_metric.get("tolerance_id")
        if tolerance_id not in rows:
            continue
        require(tolerance_id not in observed, f"duplicate numerical evidence: {tolerance_id}")
        observed[tolerance_id] = {
            "label": envelope["label"],
            "metrics": validate_metric(raw_metric, rows[tolerance_id], f"line {line_number}"),
        }
    missing = sorted(set(rows) - set(observed))
    require(not missing, f"missing Metal op/state numerical evidence: {missing}")
    return {
        "required_test_count": len(required_tests),
        "required_tests": required_tests,
        "row_count": len(rows),
        "rows": [observed[tolerance_id] for tolerance_id in sorted(observed)],
    }


def test_binary_from_log(log: str) -> Path:
    matches = re.findall(r"Running unittests .* \(([^\n]+/ferrum_kernels-[0-9a-f]+)\)", log)
    require(len(matches) == 1, "cargo log does not identify exactly one ferrum-kernels test binary")
    candidate = Path(matches[0])
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    require(candidate.is_file(), f"test binary is unavailable: {candidate}")
    return candidate.resolve()


def validate_receipt(artifact_root: Path, *, git_revision: str = "HEAD") -> dict[str, Any]:
    artifact_root = artifact_root.expanduser().resolve()
    receipt_path = artifact_root / "metal-op-numerics.json"
    receipt = load_json(receipt_path)
    require(isinstance(receipt, dict), "metal-op-numerics.json must be an object")
    require(receipt.get("schema_version") == 1 and receipt.get("status") == "pass", "receipt is not PASS schema v1")
    source_git_sha = receipt.get("source_git_sha")
    require(isinstance(source_git_sha, str) and GIT_SHA_RE.fullmatch(source_git_sha) is not None, "receipt source SHA is invalid")
    catalog, provenance = tolerances.load_catalog_from_git(git_revision, None)
    summary = tolerances.validate_catalog_document(catalog, require_complete=True)
    tolerances.validate_catalog_provenance(catalog, provenance["commit"])
    require(source_git_sha == provenance["commit"], "receipt source SHA is stale")
    require(receipt.get("catalog_git_blob_sha") == provenance["git_blob_sha"], "receipt catalog blob is stale")
    stdout_path = artifact_root / "cargo-test.stdout.log"
    stderr_path = artifact_root / "cargo-test.stderr.log"
    require(receipt.get("stdout_sha256") == sha256_file(stdout_path), "stdout log SHA differs")
    require(receipt.get("stderr_sha256") == sha256_file(stderr_path), "stderr log SHA differs")
    parsed = parse_log(stdout_path.read_text(encoding="utf-8") + "\n" + stderr_path.read_text(encoding="utf-8"), catalog)
    require(receipt.get("summary") == parsed, "receipt summary differs from logs")
    require(receipt.get("catalog_row_count") == summary["row_count"], "catalog row count differs")
    binary = receipt.get("test_binary")
    require(isinstance(binary, dict), "test binary receipt is missing")
    binary_path = Path(binary.get("path", ""))
    require(binary_path.is_file(), "recorded test binary is unavailable")
    require(binary.get("sha256") == sha256_file(binary_path), "test binary SHA differs")
    return receipt


def collect(out: Path, *, timeout_seconds: int) -> dict[str, Any]:
    out = out.expanduser().resolve()
    require(not out.exists() or not any(out.iterdir()), "output directory is not empty")
    require(not git_text("status", "--short"), "source worktree must be clean")
    source_git_sha = git_text("rev-parse", "HEAD")
    catalog, provenance = tolerances.load_catalog_from_git(source_git_sha, None)
    summary = tolerances.validate_catalog_document(catalog, require_complete=True)
    tolerances.validate_catalog_provenance(catalog, source_git_sha)
    command = [
        "cargo",
        "test",
        "-p",
        "ferrum-kernels",
        "--features",
        "metal",
        "--lib",
        "on_real_metal",
        "--",
        "--nocapture",
        "--test-threads=1",
    ]
    env = os.environ.copy()
    env["RUST_TEST_THREADS"] = "1"
    started_at = datetime.now(timezone.utc).astimezone().isoformat()
    try:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise EvidenceError(f"Metal op numerics exceeded {timeout_seconds}s") from error
    completed_at = datetime.now(timezone.utc).astimezone().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    stdout_path = out / "cargo-test.stdout.log"
    stderr_path = out / "cargo-test.stderr.log"
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    require(process.returncode == 0, f"cargo test failed with exit {process.returncode}")
    combined = process.stdout + "\n" + process.stderr
    parsed = parse_log(combined, catalog)
    binary_path = test_binary_from_log(combined)
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "source_git_sha": source_git_sha,
        "source_dirty": False,
        "catalog_git_blob_sha": provenance["git_blob_sha"],
        "catalog_row_count": summary["row_count"],
        "command": command,
        "timeout_seconds": timeout_seconds,
        "started_at": started_at,
        "completed_at": completed_at,
        "exit_code": process.returncode,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
        "test_binary": {"path": str(binary_path), "sha256": sha256_file(binary_path)},
        "summary": parsed,
    }
    write_json(out / "metal-op-numerics.json", receipt)
    validate_receipt(out, git_revision=source_git_sha)
    return receipt


def fixture_metric(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "tolerance_id": row["tolerance_id"],
        "row_fingerprint": row["row_fingerprint"],
        "element_count": 2,
        "shape": [2],
        "logical_dtype": row["dtype"],
        "oracle_precision": "fp32",
        "actual_f32_sha256": "a" * 64,
        "expected_f32_sha256": "b" * 64,
        "actual_nan_count": 0,
        "actual_inf_count": 0,
        "expected_nan_count": 0,
        "expected_inf_count": 0,
        "max_abs": 0.0,
        "max_relative_error": 0.0,
        "max_relative_error_denominator_floor": 1.0e-12,
        "relative_l2": 0.0,
        "cosine": 1.0,
    }


def self_test() -> None:
    catalog, _ = tolerances.load_catalog_from_git("HEAD", None)
    tolerances.validate_catalog_document(catalog, require_complete=True)
    rows = operation_rows(catalog)
    tests = sorted({row["basis"]["test_name"] for row in rows.values()})
    test_lines = [f"test module::{name} ... ok" for name in tests]
    metric_lines = [
        METRICS_PREFIX + json.dumps({"label": row["tolerance_id"], "metrics": fixture_metric(row)}, sort_keys=True)
        for row in rows.values()
    ]
    log = "\n".join([*test_lines, *metric_lines, "test result: ok. 7 passed; 0 failed"])
    parsed = parse_log(log, catalog)
    require(parsed["row_count"] == 27 and parsed["required_test_count"] == 7, "selftest summary differs")

    def rejects(candidate: str, marker: str) -> None:
        try:
            parse_log(candidate, catalog)
        except EvidenceError as error:
            require(marker.lower() in str(error).lower(), f"wrong rejection: {error}")
            return
        raise EvidenceError(f"selftest mutation unexpectedly passed: {marker}")

    rejects(log.replace(test_lines[0] + "\n", "", 1), "required Metal test")
    rejects(log.replace(metric_lines[0] + "\n", "", 1), "missing Metal op/state")
    wrong = json.loads(metric_lines[0][len(METRICS_PREFIX) :])
    wrong["metrics"]["row_fingerprint"] = "0" * 64
    rejects(log.replace(metric_lines[0], METRICS_PREFIX + json.dumps(wrong, sort_keys=True), 1), "fingerprint")
    rejects(log + "\nno Metal device; skipping conformance", "skip")
    print(SELFTEST_PASS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--out", type=Path, required=True)
    collect_parser.add_argument("--timeout-seconds", type=int, default=1800)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("artifact_root", type=Path)
    validate_parser.add_argument("--git-revision", default="HEAD")
    subparsers.add_parser("self-test")
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
            return 0
        if args.command == "collect":
            require(args.timeout_seconds > 0, "timeout must be positive")
            collect(args.out, timeout_seconds=args.timeout_seconds)
            print(f"{PASS_PREFIX}: {args.out.expanduser().resolve()}")
            return 0
        validate_receipt(args.artifact_root, git_revision=args.git_revision)
        print(f"{PASS_PREFIX}: {args.artifact_root.expanduser().resolve()}")
        return 0
    except (EvidenceError, tolerances.CatalogError, OSError, ValueError) as error:
        print(f"{FAIL_PREFIX}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
