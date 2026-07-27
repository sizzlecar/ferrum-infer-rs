#!/usr/bin/env python3
"""Validate the formal Qwen3.5 layer-3 full-attention numerical artifact."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import runtime_vnext_numerical_tolerances as tolerances
import runtime_vnext_qwen35_layer_reference_gate as base


PASS_PREFIX = "RUNTIME VNEXT QWEN35 FULL ATTENTION NUMERICS PASS"
SELF_TEST_PASS = "RUNTIME VNEXT QWEN35 FULL ATTENTION NUMERICS SELF-TEST PASS"
TOLERANCE_ID = (
    "runtime-vnext.metal.qwen35-4b.full-attention.v2.layer."
    "fp16.gguf-q4-k-m.tokens-24"
)
EXPECTED_MISSING = ["checkpoint.full_model", "checkpoint.full_vocab_logits"]
EXPECTED_MODEL_SHA256 = "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
EXPECTED_TOKEN_SHA256 = "8276dc19eb8a689a640328eb30be55725913ffd9aa291b01f040cbb9543e5e6f"
EXPECTED_SOURCE_SHA256 = "0826ebd5387f4b442d73033c339c7bb54f22ff237cc654bcbcea96385baceefa"
EXPECTED_COMMON_SOURCE_SHA256 = (
    "13bfdefbe247b2b0f6b5fe4b5666af5cf8e7c26d5c84e4ec9cedec2bfbbd5137"
)
EXPECTED_SHAPE = [24, 2560]
EXPECTED_ELEMENTS = 24 * 2560
INPUT_VALUE_ID = "value.layer.2.output"
OUTPUT_VALUE_ID = "value.layer.3.attention"
SHA_RECORD_RE = re.compile(r"^[0-9a-f]{64}(?:\s+.+)?$")
STAGE_SHA_FIELDS = frozenset(
    {
        "input_norm_f32_sha256",
        "query_raw_f32_sha256",
        "key_raw_f32_sha256",
        "value_raw_f32_sha256",
        "query_rope_f32_sha256",
        "key_rope_f32_sha256",
        "context_gated_f32_sha256",
        "projected_f32_sha256",
        "residual_f32_sha256",
    }
)


def validate_sha_map(value: Any, label: str) -> dict[str, str]:
    values = base.exact_fields(value, STAGE_SHA_FIELDS, label)
    for name, digest in values.items():
        base.require(
            isinstance(digest, str) and base.SHA256_RE.fullmatch(digest) is not None,
            f"{label}.{name} is not SHA256",
        )
    return values


def validate_cross_metrics(value: Any, label: str) -> dict[str, float]:
    base.require(isinstance(value, dict), f"{label} must be an object")
    base.require(
        value.get("element_count") == EXPECTED_ELEMENTS,
        f"{label}.element_count differs",
    )
    for name in (
        "actual_nan_count",
        "actual_inf_count",
        "expected_nan_count",
        "expected_inf_count",
    ):
        base.require(value.get(name) == 0, f"{label}.{name} != 0")
    base.require(
        value.get("max_relative_error_denominator_floor") == 1.0e-12,
        f"{label} relative-error denominator floor differs",
    )
    metrics = {
        name: base.finite_number(value.get(name), f"{label}.{name}")
        for name in ("max_abs", "max_relative_error", "relative_l2", "cosine")
    }
    base.require(metrics["max_abs"] >= 0.0, f"{label}.max_abs is negative")
    base.require(metrics["relative_l2"] >= 0.0, f"{label}.relative_l2 is negative")
    base.require(-1.0 <= metrics["cosine"] <= 1.0, f"{label}.cosine is invalid")
    return metrics


def validate_model(artifact_dir: Path, model: Any) -> None:
    base.require(
        isinstance(model, dict) and model.get("sha256") == EXPECTED_MODEL_SHA256,
        "model SHA256 differs from the catalog fixture",
    )
    base.require(
        model.get("format") == "GGUF Q4_K_M" and model.get("tensor_count") == 426,
        "model format or tensor inventory count differs",
    )
    base.require(
        model.get("byte_count") == 2740937888,
        "model byte count differs from the reviewed fixture",
    )
    base.require(
        model.get("hugging_face_snapshot")
        == {
            "repository": "unsloth/Qwen3.5-4B-GGUF",
            "revision": "e87f176479d0855a907a41277aca2f8ee7a09523",
        },
        "Hugging Face snapshot identity differs from the reviewed fixture",
    )
    for file_key, sha_key in (
        ("metadata_file", "metadata_sha256"),
        ("tensor_inventory_file", "tensor_inventory_sha256"),
    ):
        path = base.safe_child(artifact_dir, model.get(file_key), f"model.{file_key}")
        base.require(
            base.sha256_file(path) == model.get(sha_key),
            f"model {file_key} SHA256 mismatch",
        )


def validate_fixture(fixture: Any, row: dict[str, Any]) -> None:
    base.require(isinstance(fixture, dict), "report.fixture must be an object")
    token_ids = fixture.get("token_ids")
    base.require(
        isinstance(token_ids, list) and len(token_ids) == EXPECTED_SHAPE[0],
        "fixture token IDs are incomplete",
    )
    base.require(
        all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 <= value < 2**32
            for value in token_ids
        ),
        "fixture token IDs contain invalid values",
    )
    token_sha = hashlib.sha256(
        b"".join(int(value).to_bytes(4, "little") for value in token_ids)
    ).hexdigest()
    base.require(
        token_sha == EXPECTED_TOKEN_SHA256
        and fixture.get("token_sequence_sha256") == EXPECTED_TOKEN_SHA256,
        "fixture token sequence differs from catalog row",
    )
    shape_domain = {
        "fixture_id": fixture.get("fixture_id"),
        "dimensions": {
            "head_dim": fixture.get("head_dim"),
            "hidden_size": EXPECTED_SHAPE[1],
            "key_value_heads": fixture.get("key_value_heads"),
            "query_heads": fixture.get("query_heads"),
            "rope_dim": fixture.get("rope_dim"),
            "tokens": EXPECTED_SHAPE[0],
        },
        "semantics": {
            "causal": fixture.get("causal"),
            "input_value_id": fixture.get("input_value_id"),
            "layer_index": fixture.get("layer_index"),
            "model_sha256": EXPECTED_MODEL_SHA256,
            "output_gate": fixture.get("output_gate"),
            "output_value_id": fixture.get("output_value_id"),
            "prompt_token_sequence_sha256": fixture.get("token_sequence_sha256"),
            "rope_interleaved": fixture.get("rope_interleaved"),
            "rope_theta": fixture.get("rope_theta"),
        },
    }
    base.require(fixture.get("shape") == EXPECTED_SHAPE, "fixture shape differs")
    base.require(shape_domain == row["shape_domain"], "fixture semantics differ from row")


def validate_actual(actual: Any) -> dict[str, Path]:
    actual = base.exact_fields(
        actual,
        frozenset(
            {
                "artifact_root",
                "plan_file",
                "plan_sha256",
                "wave_file",
                "wave_sha256",
                "git_sha",
                "git_status",
                "tracked_diff_sha256",
                "binary_sha256_record",
                "checkpoints",
            }
        ),
        "report.actual",
    )
    artifact_root = Path(str(actual.get("artifact_root", ""))).resolve()
    capture_root = artifact_root / "capture"
    base.require(capture_root.is_dir(), "actual capture artifact root is unavailable")
    for file_key, sha_key in (("plan_file", "plan_sha256"), ("wave_file", "wave_sha256")):
        path = base.safe_child(capture_root, actual.get(file_key), f"actual.{file_key}")
        base.require(base.sha256_file(path) == actual.get(sha_key), f"actual {file_key} SHA mismatch")
    base.require(
        actual.get("tracked_diff_sha256") == base.EMPTY_SHA256,
        "actual capture contains tracked source changes",
    )
    binary_record = actual.get("binary_sha256_record")
    base.require(
        isinstance(binary_record, str) and SHA_RECORD_RE.fullmatch(binary_record) is not None,
        "actual capture binary SHA256 record is invalid",
    )
    status = actual.get("git_status")
    base.require(isinstance(status, str), "actual capture Git status is absent")
    base.require(
        all(line.startswith("?? ") for line in status.splitlines()),
        "actual capture Git status contains tracked changes",
    )
    base.require_git_commit(actual.get("git_sha"), "actual capture commit")

    checkpoints = base.exact_fields(
        actual.get("checkpoints"),
        frozenset({INPUT_VALUE_ID, OUTPUT_VALUE_ID}),
        "actual.checkpoints",
    )
    paths: dict[str, Path] = {}
    for value_id, checkpoint in checkpoints.items():
        checkpoint = base.exact_fields(
            checkpoint,
            frozenset({"logical_dtype", "logical_shape", "raw_file", "raw_sha256"}),
            f"actual.checkpoints.{value_id}",
        )
        base.require(
            checkpoint["logical_dtype"] == "fp16"
            and checkpoint["logical_shape"] == EXPECTED_SHAPE,
            f"{value_id} dtype/shape differs",
        )
        raw = base.safe_child(capture_root, checkpoint["raw_file"], f"{value_id}.raw_file")
        base.require(raw.stat().st_size == EXPECTED_ELEMENTS * 2, f"{value_id} byte count differs")
        base.require(base.sha256_file(raw) == checkpoint["raw_sha256"], f"{value_id} SHA mismatch")
        paths[value_id] = raw
    return paths


def validate_cross_validation(value: Any) -> dict[str, Any]:
    value = base.exact_fields(
        value,
        frozenset(
            {
                "role",
                "provenance",
                "input_ferrum_vs_llama",
                "residual_llama_vs_its_fp32_oracle",
                "oracle_stage_f32_sha256",
            }
        ),
        "llama_cpp_cross_validation",
    )
    base.require(
        value["role"] == "same-GGUF quantized implementation cross-validation only",
        "llama.cpp cross-validation role is invalid",
    )
    provenance = base.exact_fields(
        value["provenance"],
        frozenset(
            {
                "artifact_root",
                "manifest_sha256",
                "checkpoints",
                "extractor-binary-sha256",
                "extractor-source-sha256",
                "llama-runtime-version",
            }
        ),
        "llama_cpp_cross_validation.provenance",
    )
    artifact_root = Path(str(provenance["artifact_root"])).resolve()
    checkpoint_root = artifact_root / "checkpoints"
    manifest = checkpoint_root / "checkpoint-manifest.json"
    base.require(manifest.is_file() and not manifest.is_symlink(), "llama manifest is unavailable")
    base.require(base.sha256_file(manifest) == provenance["manifest_sha256"], "llama manifest SHA mismatch")
    for label in ("extractor-binary-sha256", "extractor-source-sha256"):
        record = provenance[label]
        base.require(
            isinstance(record, str) and SHA_RECORD_RE.fullmatch(record) is not None,
            f"llama {label} is invalid",
        )
    base.require(
        isinstance(provenance["llama-runtime-version"], str)
        and "version:" in provenance["llama-runtime-version"],
        "llama runtime version is absent",
    )
    checkpoints = base.exact_fields(
        provenance["checkpoints"],
        frozenset({"l_out-2", "attn_residual-3"}),
        "llama_cpp_cross_validation.provenance.checkpoints",
    )
    for name, checkpoint in checkpoints.items():
        checkpoint = base.exact_fields(
            checkpoint,
            frozenset({"logical_dtype", "logical_shape", "raw_file", "raw_sha256"}),
            f"llama.checkpoints.{name}",
        )
        base.require(
            checkpoint["logical_dtype"] == "fp32"
            and checkpoint["logical_shape"] == EXPECTED_SHAPE,
            f"llama {name} dtype/shape differs",
        )
        raw = base.safe_child(checkpoint_root, checkpoint["raw_file"], f"llama.{name}.raw_file")
        base.require(raw.stat().st_size == EXPECTED_ELEMENTS * 4, f"llama {name} byte count differs")
        base.require(base.sha256_file(raw) == checkpoint["raw_sha256"], f"llama {name} SHA mismatch")
    validate_sha_map(value["oracle_stage_f32_sha256"], "llama oracle stages")
    return {
        "input_ferrum_vs_llama": validate_cross_metrics(
            value["input_ferrum_vs_llama"], "llama input cross-validation"
        ),
        "residual_llama_vs_its_fp32_oracle": validate_cross_metrics(
            value["residual_llama_vs_its_fp32_oracle"],
            "llama residual cross-validation",
        ),
    }


def validate_report(artifact_dir: Path, report: Any, row: dict[str, Any]) -> dict[str, Any]:
    report = base.exact_fields(
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
    base.forbid_artifact_tolerances(report)
    base.require(
        report["schema_version"] == 1 and report["status"] == "measured",
        "report is not a measured schema-v1 artifact",
    )

    oracle = report["oracle"]
    base.require(isinstance(oracle, dict), "report.oracle must be an object")
    base.require(oracle.get("identity") == row["oracle_identity"], "oracle identity differs")
    base.require(oracle.get("precision") == row["oracle_precision"], "oracle precision differs")
    base.require(oracle.get("source_path") == row["basis"]["source_path"], "oracle source path differs")
    source = oracle.get("ferrum_source")
    base.require(
        isinstance(source, dict) and source.get("tracked_dirty") is False,
        "oracle source worktree was dirty",
    )
    base.require(source.get("git_sha") == row["source_commit"], "oracle source commit differs")
    source_payload = base.git_bytes(row["source_commit"], row["basis"]["source_path"])
    base.require(hashlib.sha256(source_payload).hexdigest() == EXPECTED_SOURCE_SHA256, "reviewed oracle source hash differs")
    base.require(oracle.get("source_sha256") == EXPECTED_SOURCE_SHA256, "artifact oracle source hash differs")
    common_path = oracle.get("common_source_path")
    base.require(common_path == "scripts/release/qwen35_gguf_linear_attention_reference.py", "common source path differs")
    common_payload = base.git_bytes(row["source_commit"], common_path)
    base.require(hashlib.sha256(common_payload).hexdigest() == EXPECTED_COMMON_SOURCE_SHA256, "reviewed common source hash differs")
    base.require(oracle.get("common_source_sha256") == EXPECTED_COMMON_SOURCE_SHA256, "artifact common source hash differs")
    llama_source = oracle.get("llama_cpp_gguf_py_source")
    base.require(
        isinstance(llama_source, dict)
        and llama_source.get("tracked_dirty") is False
        and isinstance(llama_source.get("git_sha"), str)
        and base.GIT_SHA_RE.fullmatch(llama_source["git_sha"]) is not None,
        "gguf-py source provenance is invalid",
    )
    base.require_git_commit(source.get("git_sha"), "oracle source commit")

    validate_model(artifact_dir, report["model"])
    validate_fixture(report["fixture"], row)
    actual_paths = validate_actual(report["actual"])

    reference = base.exact_fields(
        report["reference"],
        frozenset({"logical_dtype", "logical_shape", "raw_file", "raw_sha256", "stage_f32_sha256"}),
        "report.reference",
    )
    base.require(
        reference["logical_dtype"] == "fp32" and reference["logical_shape"] == EXPECTED_SHAPE,
        "reference dtype/shape differs",
    )
    raw_reference = base.safe_child(artifact_dir, reference["raw_file"], "reference.raw_file")
    base.require(raw_reference.stat().st_size == EXPECTED_ELEMENTS * 4, "reference byte count differs")
    reference_sha = base.sha256_file(raw_reference)
    base.require(reference_sha == reference["raw_sha256"], "reference SHA mismatch")
    stages = validate_sha_map(reference["stage_f32_sha256"], "reference stages")
    base.require(stages["residual_f32_sha256"] == reference_sha, "reference residual stage SHA differs")

    metrics = base.validate_metrics(report["metrics"], row)
    base.require(report["metrics"]["expected_f32_sha256"] == reference_sha, "metric/reference SHA differs")
    base.require(
        report["metrics"]["actual_f32_sha256"]
        == base.f16_as_f32_sha256(actual_paths[OUTPUT_VALUE_ID], EXPECTED_ELEMENTS),
        "metric/actual FP32 SHA differs",
    )
    cross_validation = validate_cross_validation(report["llama_cpp_cross_validation"])
    return {
        "metrics": metrics,
        "input_raw_sha256": base.sha256_file(actual_paths[INPUT_VALUE_ID]),
        "actual_raw_sha256": base.sha256_file(actual_paths[OUTPUT_VALUE_ID]),
        "reference_raw_sha256": reference_sha,
        "llama_cpp_cross_validation": cross_validation,
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
        "max_abs": 0.0021,
        "max_relative_error": 12.0,
        "max_relative_error_denominator_floor": 1.0e-12,
        "relative_l2": 0.0003,
        "cosine": 0.99999,
    }
    base.validate_metrics(metrics, row)
    validate_sha_map({name: "a" * 64 for name in STAGE_SHA_FIELDS}, "stages")
    rejected = dict(metrics)
    rejected["max_abs"] = 0.02
    try:
        base.validate_metrics(rejected, row)
    except base.GateError as error:
        base.require("max_abs" in str(error), "wrong numerical rejection")
    else:
        raise base.GateError("out-of-bound full-attention metric unexpectedly passed")
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
        base.require(args.artifact_dir is not None and args.out is not None, "artifact_dir and --out are required")
        artifact_dir = Path(args.artifact_dir).expanduser().resolve()
        out_dir = Path(args.out).expanduser().resolve()
        base.require(artifact_dir.is_dir(), "reference artifact directory is unavailable")
        base.require(not out_dir.exists() or out_dir.is_dir(), "--out is not a directory")
        base.require(not out_dir.exists() or not any(out_dir.iterdir()), "--out directory is not empty")

        catalog, provenance = tolerances.load_catalog_from_git(
            args.git_revision, args.expected_catalog_blob_sha
        )
        summary = tolerances.validate_catalog_document(catalog)
        tolerances.validate_catalog_provenance(catalog, provenance["commit"])
        compared = tolerances.validate_no_widening_from_revision(catalog, provenance["commit"])
        base.require(summary["missing_required_coverage"] == EXPECTED_MISSING, "catalog G08A coverage gap differs")
        rows = [row for row in catalog["rows"] if row["tolerance_id"] == TOLERANCE_ID]
        base.require(len(rows) == 1, "catalog does not contain exactly one full-attention row")
        row = rows[0]
        base.require(row["coverage_markers"] == ["layer.full_attention"], "full-attention markers differ")
        base.require(row["row_fingerprint"] == tolerances.row_fingerprint(row), "full-attention row fingerprint differs")

        report_path = artifact_dir / "report.json"
        validation = validate_report(artifact_dir, base.load_json(report_path), row)
        out_dir.mkdir(parents=True, exist_ok=True)
        base.write_json(
            out_dir / "qwen35-full-attention.gate.json",
            {
                "schema_version": 1,
                "status": "pass",
                "reference_artifact": str(artifact_dir),
                "reference_report_sha256": base.sha256_file(report_path),
                "catalog_commit": provenance["commit"],
                "catalog_git_blob_sha": provenance["git_blob_sha"],
                "compared_catalog_commits": compared,
                "tolerance_id": TOLERANCE_ID,
                "row_fingerprint": row["row_fingerprint"],
                **validation,
            },
        )
    except (base.GateError, tolerances.CatalogError) as error:
        print(f"RUNTIME VNEXT QWEN35 FULL ATTENTION NUMERICS FAIL: {error}", file=sys.stderr)
        return 1
    print(f"{PASS_PREFIX}: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
