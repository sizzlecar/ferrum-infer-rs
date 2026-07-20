#!/usr/bin/env python3
"""Validate the formal Qwen3.5 layer-0 linear-attention numerical artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path, PurePath
from typing import Any

import runtime_vnext_numerical_tolerances as tolerances


PASS_PREFIX = "RUNTIME VNEXT QWEN35 LINEAR ATTENTION NUMERICS PASS"
SELF_TEST_PASS = "RUNTIME VNEXT QWEN35 LINEAR ATTENTION NUMERICS SELF-TEST PASS"
TOLERANCE_ID = (
    "runtime-vnext.metal.qwen35-4b.linear-attention.v4.layer."
    "fp16.gguf-q4-k-m.tokens-24"
)
EXPECTED_MISSING = [
    "checkpoint.full_model",
    "checkpoint.full_vocab_logits",
    "layer.full_attention",
]
EXPECTED_MODEL_SHA256 = "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
EXPECTED_TOKEN_SHA256 = "8276dc19eb8a689a640328eb30be55725913ffd9aa291b01f040cbb9543e5e6f"
EXPECTED_SOURCE_SHA256 = "13bfdefbe247b2b0f6b5fe4b5666af5cf8e7c26d5c84e4ec9cedec2bfbbd5137"
EXPECTED_SHAPE = [24, 2560]
EXPECTED_ELEMENTS = 24 * 2560
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {
        "bounds",
        "cosine_min",
        "max_abs_max",
        "relative_l2_max",
        "row_fingerprint",
        "tolerance",
        "tolerance_id",
        "tolerances",
    }
)


class GateError(RuntimeError):
    """The numerical artifact is incomplete, stale, or outside its catalog row."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise GateError(f"non-finite JSON constant is forbidden: {value}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except GateError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"cannot load {path}: {error}") from error


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise GateError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def safe_child(root: Path, raw_name: Any, label: str) -> Path:
    require(isinstance(raw_name, str) and raw_name == raw_name.strip(),
            f"{label} must be a trimmed string")
    pure = PurePath(raw_name)
    require(bool(raw_name) and not pure.is_absolute() and ".." not in pure.parts,
            f"{label} must remain within its artifact directory")
    path = root / pure
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    return path


def exact_fields(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    require(not expected - actual, f"{label} is missing fields: {sorted(expected - actual)}")
    require(not actual - expected, f"{label} has unknown fields: {sorted(actual - expected)}")
    return value


def forbid_artifact_tolerances(value: Any, label: str = "report") -> None:
    if isinstance(value, dict):
        forbidden = sorted(set(value) & FORBIDDEN_ARTIFACT_KEYS)
        require(not forbidden, f"{label} carries forbidden artifact-local tolerance: {forbidden}")
        for key, item in value.items():
            forbid_artifact_tolerances(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            forbid_artifact_tolerances(item, f"{label}[{index}]")


def finite_number(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label} must be numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} must be finite")
    return number


def git_bytes(revision: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=tolerances.REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(completed.returncode == 0,
            f"cannot load oracle source {revision}:{path}: "
            f"{completed.stderr.decode(errors='replace').strip()}")
    return completed.stdout


def require_git_commit(commit: Any, label: str) -> str:
    require(isinstance(commit, str) and GIT_SHA_RE.fullmatch(commit) is not None,
            f"{label} must be a full Git SHA")
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=tolerances.REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(completed.returncode == 0, f"{label} is not present in the repository")
    return commit


def f16_as_f32_sha256(path: Path, expected_elements: int) -> str:
    payload = path.read_bytes()
    require(len(payload) == expected_elements * 2, "actual FP16 checkpoint byte count differs")
    digest = hashlib.sha256()
    for (value,) in struct.iter_unpack("<e", payload):
        require(math.isfinite(value), "actual FP16 checkpoint contains NaN or Inf")
        digest.update(struct.pack("<f", value))
    return digest.hexdigest()


def validate_metrics(metrics: Any, row: dict[str, Any]) -> dict[str, float]:
    require(isinstance(metrics, dict), "report.metrics must be an object")
    require(metrics.get("shape") == EXPECTED_SHAPE, "metric shape differs from catalog fixture")
    require(metrics.get("element_count") == EXPECTED_ELEMENTS,
            "metric element count differs from catalog fixture")
    require(metrics.get("actual_logical_dtype") == row["dtype"],
            "metric dtype differs from catalog row")
    require(metrics.get("oracle_precision") == row["oracle_precision"],
            "metric oracle precision differs from catalog row")
    for name in (
        "actual_nan_count",
        "actual_inf_count",
        "expected_nan_count",
        "expected_inf_count",
    ):
        require(metrics.get(name) == 0, f"metric invariant failed: {name} != 0")
    for name in ("actual_f32_sha256", "expected_f32_sha256"):
        require(isinstance(metrics.get(name), str)
                and SHA256_RE.fullmatch(metrics[name]) is not None,
                f"metric {name} is not SHA256")

    max_abs = finite_number(metrics.get("max_abs"), "metrics.max_abs")
    relative_l2 = finite_number(metrics.get("relative_l2"), "metrics.relative_l2")
    cosine = finite_number(metrics.get("cosine"), "metrics.cosine")
    max_relative_error = finite_number(
        metrics.get("max_relative_error"), "metrics.max_relative_error"
    )
    require(metrics.get("max_relative_error_denominator_floor") == 1.0e-12,
            "metric relative-error denominator floor differs")
    bounds = row["bounds"]
    require(max_abs <= bounds["max_abs_max"],
            f"max_abs {max_abs} exceeds {bounds['max_abs_max']}")
    require(relative_l2 <= bounds["relative_l2_max"],
            f"relative_l2 {relative_l2} exceeds {bounds['relative_l2_max']}")
    require(cosine >= bounds["cosine_min"],
            f"cosine {cosine} is below {bounds['cosine_min']}")
    return {
        "max_abs": max_abs,
        "max_relative_error": max_relative_error,
        "relative_l2": relative_l2,
        "cosine": cosine,
    }


def validate_report(
    artifact_dir: Path,
    report: Any,
    row: dict[str, Any],
) -> dict[str, Any]:
    report = exact_fields(
        report,
        frozenset(
            {
                "schema_version",
                "status",
                "oracle",
                "model",
                "fixture",
                "actual",
                "reference",
                "metrics",
                "llama_cpp_cross_validation",
                "invocation",
            }
        ),
        "report",
    )
    forbid_artifact_tolerances(report)
    require(report["schema_version"] == 1 and report["status"] == "measured",
            "report is not a measured schema-v1 artifact")

    oracle = report["oracle"]
    require(isinstance(oracle, dict), "report.oracle must be an object")
    require(oracle.get("identity") == row["oracle_identity"],
            "oracle identity differs from catalog row")
    require(oracle.get("precision") == row["oracle_precision"],
            "oracle precision differs from catalog row")
    require(oracle.get("source_path") == row["basis"]["source_path"],
            "oracle source path differs from catalog row")
    source = oracle.get("ferrum_source")
    require(isinstance(source, dict) and source.get("tracked_dirty") is False,
            "oracle source worktree was dirty")
    require(source.get("git_sha") == row["source_commit"],
            "oracle source commit differs from catalog row")
    source_payload = git_bytes(row["source_commit"], row["basis"]["source_path"])
    require(hashlib.sha256(source_payload).hexdigest() == EXPECTED_SOURCE_SHA256,
            "catalog oracle source hash differs from the reviewed source")
    require(oracle.get("source_sha256") == EXPECTED_SOURCE_SHA256,
            "artifact oracle source hash differs from the reviewed source")
    llama_source = oracle.get("llama_cpp_gguf_py_source")
    require(isinstance(llama_source, dict) and llama_source.get("tracked_dirty") is False,
            "gguf-py source worktree was dirty")
    require_git_commit(source.get("git_sha"), "oracle source commit")

    model = report["model"]
    require(isinstance(model, dict) and model.get("sha256") == EXPECTED_MODEL_SHA256,
            "model SHA256 differs from the catalog fixture")
    require(model.get("format") == "GGUF Q4_K_M" and model.get("tensor_count") == 426,
            "model format or tensor inventory count differs")
    require(model.get("byte_count") == 2740937888,
            "model byte count differs from the reviewed fixture")
    snapshot = model.get("hugging_face_snapshot")
    require(
        snapshot == {
            "repository": "unsloth/Qwen3.5-4B-GGUF",
            "revision": "e87f176479d0855a907a41277aca2f8ee7a09523",
        },
        "Hugging Face snapshot identity differs from the reviewed fixture",
    )
    for file_key, sha_key in (
        ("metadata_file", "metadata_sha256"),
        ("tensor_inventory_file", "tensor_inventory_sha256"),
    ):
        path = safe_child(artifact_dir, model.get(file_key), f"model.{file_key}")
        require(sha256_file(path) == model.get(sha_key),
                f"model {file_key} SHA256 mismatch")

    fixture = report["fixture"]
    require(isinstance(fixture, dict), "report.fixture must be an object")
    require(fixture.get("fixture_id") == row["shape_domain"]["fixture_id"],
            "fixture id differs from catalog row")
    require(fixture.get("shape") == EXPECTED_SHAPE and fixture.get("layer_index") == 0,
            "fixture shape/layer differs from catalog row")
    require(fixture.get("decay_parameterization") == "negative_rate"
            and fixture.get("value_head_mapping") == "interleaved_by_key_head",
            "fixture GDN semantics differ from catalog row")
    token_ids = fixture.get("token_ids")
    require(isinstance(token_ids, list) and len(token_ids) == EXPECTED_SHAPE[0],
            "fixture token IDs are incomplete")
    require(
        all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value < 2**32
            for value in token_ids
        ),
        "fixture token IDs contain invalid values",
    )
    token_payload = b"".join(
        int(value).to_bytes(4, "little") for value in token_ids
    )
    token_sha = hashlib.sha256(token_payload).hexdigest()
    require(token_sha == EXPECTED_TOKEN_SHA256
            and fixture.get("token_sequence_sha256") == EXPECTED_TOKEN_SHA256,
            "fixture token sequence differs from catalog row")

    actual = report["actual"]
    require(isinstance(actual, dict), "report.actual must be an object")
    require(actual.get("value_id") == "value.layer.0.attention",
            "actual checkpoint value id differs")
    require(actual.get("logical_dtype") == "fp16"
            and actual.get("logical_shape") == EXPECTED_SHAPE,
            "actual checkpoint dtype/shape differs")
    actual_root = Path(str(actual.get("artifact_root", ""))).resolve()
    require(actual_root.is_dir(), "actual capture artifact root is unavailable")
    raw_actual = safe_child(actual_root / "capture", actual.get("raw_file"),
                            "actual.raw_file")
    require(sha256_file(raw_actual) == actual.get("raw_sha256"),
            "actual raw checkpoint SHA256 mismatch")
    require(actual.get("tracked_diff_sha256") == EMPTY_SHA256,
            "actual capture contains tracked source changes")
    binary_record = actual.get("binary_sha256_record")
    require(isinstance(binary_record, str)
            and SHA256_RE.fullmatch(binary_record.split(maxsplit=1)[0]) is not None,
            "actual capture binary SHA256 record is invalid")
    status = actual.get("git_status")
    require(isinstance(status, str), "actual capture Git status is absent")
    require(all(line.startswith("?? ") for line in status.splitlines()),
            "actual capture Git status contains tracked changes")
    require_git_commit(actual.get("git_sha"), "actual capture commit")

    reference = report["reference"]
    require(isinstance(reference, dict), "report.reference must be an object")
    require(reference.get("logical_dtype") == "fp32"
            and reference.get("logical_shape") == EXPECTED_SHAPE,
            "reference checkpoint dtype/shape differs")
    raw_reference = safe_child(artifact_dir, reference.get("raw_file"),
                               "reference.raw_file")
    require(raw_reference.stat().st_size == EXPECTED_ELEMENTS * 4,
            "reference checkpoint byte count differs")
    reference_sha = sha256_file(raw_reference)
    require(reference_sha == reference.get("raw_sha256"),
            "reference checkpoint SHA256 mismatch")

    metrics = validate_metrics(report["metrics"], row)
    require(report["metrics"]["expected_f32_sha256"] == reference_sha,
            "metric/reference FP32 SHA256 differs")
    require(
        report["metrics"]["actual_f32_sha256"]
        == f16_as_f32_sha256(raw_actual, EXPECTED_ELEMENTS),
        "metric/actual FP32 SHA256 differs",
    )

    cross_validation = report["llama_cpp_cross_validation"]
    require(isinstance(cross_validation, dict),
            "same-GGUF llama.cpp cross-validation is required")
    require(cross_validation.get("role")
            == "same-GGUF quantized implementation cross-validation only",
            "llama.cpp cross-validation role is invalid")
    cross_metrics = cross_validation.get("metrics")
    require(isinstance(cross_metrics, dict), "llama.cpp metrics are absent")
    for name in ("max_abs", "relative_l2", "cosine"):
        finite_number(cross_metrics.get(name), f"llama_cpp.metrics.{name}")
    for name in (
        "actual_nan_count",
        "actual_inf_count",
        "expected_nan_count",
        "expected_inf_count",
    ):
        require(cross_metrics.get(name) == 0,
                f"llama.cpp cross-validation invariant failed: {name}")
    return {
        "metrics": metrics,
        "actual_raw_sha256": actual["raw_sha256"],
        "reference_raw_sha256": reference_sha,
        "llama_cpp_cross_validation": {
            name: cross_metrics[name] for name in ("max_abs", "relative_l2", "cosine")
        },
    }


def self_test() -> None:
    row = {
        "dtype": "fp16",
        "oracle_precision": "fp32",
        "bounds": {"max_abs_max": 0.012, "relative_l2_max": 0.01, "cosine_min": 0.999},
    }
    metrics = {
        "shape": EXPECTED_SHAPE,
        "element_count": EXPECTED_ELEMENTS,
        "actual_logical_dtype": "fp16",
        "oracle_precision": "fp32",
        "actual_nan_count": 0,
        "actual_inf_count": 0,
        "expected_nan_count": 0,
        "expected_inf_count": 0,
        "actual_f32_sha256": "a" * 64,
        "expected_f32_sha256": "b" * 64,
        "max_abs": 0.002,
        "max_relative_error": 10.0,
        "max_relative_error_denominator_floor": 1.0e-12,
        "relative_l2": 0.0005,
        "cosine": 0.9999,
    }
    validate_metrics(metrics, row)
    rejected = dict(metrics)
    rejected["relative_l2"] = 0.02
    try:
        validate_metrics(rejected, row)
    except GateError as error:
        require("relative_l2" in str(error), "wrong numerical rejection")
    else:
        raise GateError("out-of-bound metric unexpectedly passed")
    try:
        forbid_artifact_tolerances({"metrics": metrics, "tolerance_id": "forged"})
    except GateError as error:
        require("artifact-local tolerance" in str(error), "wrong tolerance rejection")
    else:
        raise GateError("artifact-local tolerance unexpectedly passed")
    print(SELF_TEST_PASS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", nargs="?")
    parser.add_argument("--out")
    parser.add_argument("--git-revision", default="HEAD")
    parser.add_argument("--expected-catalog-blob-sha")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        require(args.artifact_dir is not None and args.out is not None,
                "artifact_dir and --out are required")
        artifact_dir = Path(args.artifact_dir).expanduser().resolve()
        out_dir = Path(args.out).expanduser().resolve()
        require(artifact_dir.is_dir(), "reference artifact directory is unavailable")
        require(not out_dir.exists() or out_dir.is_dir(), "--out is not a directory")
        require(not out_dir.exists() or not any(out_dir.iterdir()),
                "--out directory is not empty")

        catalog, catalog_provenance = tolerances.load_catalog_from_git(
            args.git_revision, args.expected_catalog_blob_sha
        )
        catalog_summary = tolerances.validate_catalog_document(catalog)
        tolerances.validate_catalog_provenance(catalog, catalog_provenance["commit"])
        compared = tolerances.validate_no_widening_from_revision(
            catalog, catalog_provenance["commit"]
        )
        require(catalog_summary["missing_required_coverage"] == EXPECTED_MISSING,
                "catalog G08A coverage gap differs from this checkpoint")
        rows = [row for row in catalog["rows"] if row["tolerance_id"] == TOLERANCE_ID]
        require(len(rows) == 1, "catalog does not contain exactly one reviewed layer row")
        row = rows[0]
        require(row["coverage_markers"]
                == ["layer.linear_attention", "quant_format.gguf_q4_k_m"],
                "layer row coverage markers differ")
        require(row["row_fingerprint"] == tolerances.row_fingerprint(row),
                "layer row fingerprint differs")

        report_path = artifact_dir / "report.json"
        report = load_json(report_path)
        validation = validate_report(artifact_dir, report, row)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "schema_version": 1,
            "status": "pass",
            "reference_artifact": str(artifact_dir),
            "reference_report_sha256": sha256_file(report_path),
            "catalog_commit": catalog_provenance["commit"],
            "catalog_git_blob_sha": catalog_provenance["git_blob_sha"],
            "compared_catalog_commits": compared,
            "tolerance_id": TOLERANCE_ID,
            "row_fingerprint": row["row_fingerprint"],
            **validation,
        }
        write_json(out_dir / "qwen35-linear-attention.gate.json", result)
    except (GateError, tolerances.CatalogError) as error:
        print(f"RUNTIME VNEXT QWEN35 LINEAR ATTENTION NUMERICS FAIL: {error}",
              file=sys.stderr)
        return 1
    print(f"{PASS_PREFIX}: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
