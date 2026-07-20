#!/usr/bin/env python3
"""Validate formal Qwen3.5 full-model and full-vocabulary numerical artifacts."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from pathlib import Path
from typing import Any

import runtime_vnext_numerical_tolerances as tolerances
import runtime_vnext_qwen35_full_attention_gate as shared


base = shared.base
PASS_PREFIX = "RUNTIME VNEXT QWEN35 MODEL NUMERICS PASS"
SELF_TEST_PASS = "RUNTIME VNEXT QWEN35 MODEL NUMERICS SELF-TEST PASS"
FULL_MODEL_TOLERANCE_ID = (
    "runtime-vnext.metal.qwen35-4b.full-model.v1.full-model."
    "fp16.gguf-q4-k-m.tokens-24"
)
LOGITS_TOLERANCE_ID = (
    "runtime-vnext.metal.qwen35-4b.full-vocab-logits.v1.full-vocab-logits."
    "fp16.gguf-q4-k-m.tokens-24"
)
EXPECTED_MODEL_SHA256 = "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
EXPECTED_TOKEN_SHA256 = "8276dc19eb8a689a640328eb30be55725913ffd9aa291b01f040cbb9543e5e6f"
EXPECTED_SOURCE_SHA256 = "cd5e4510238fc526dac07691696e38118c12a5cacc92387e341accb768caaf8f"
EXPECTED_LINEAR_COMMON_SHA256 = (
    "13bfdefbe247b2b0f6b5fe4b5666af5cf8e7c26d5c84e4ec9cedec2bfbbd5137"
)
EXPECTED_FULL_COMMON_SHA256 = (
    "0826ebd5387f4b442d73033c339c7bb54f22ff237cc654bcbcea96385baceefa"
)
EXPECTED_EXTRACTOR_SOURCE_COMMIT = "7a75cdcd6db7fb427d3b175200ef218eff364e05"
EXPECTED_EXTRACTOR_SOURCE_SHA256 = (
    "2ae2f57bfcd38d69d046e8d4dadb545f8423e72d9af46cd2836490aee2662253"
)
TOKENS = 24
HIDDEN_SIZE = 2560
LAYERS = 32
VOCABULARY_SIZE = 248320
FULL_HIDDEN_SHAPE = [TOKENS, HIDDEN_SIZE]
FULL_HIDDEN_ELEMENTS = TOKENS * HIDDEN_SIZE
LOGITS_SHAPE = [VOCABULARY_SIZE]
EMBEDDING_VALUE_ID = "value.hidden.embedding"
FINAL_HIDDEN_VALUE_ID = "value.output.final_hidden"
LOGITS_VALUE_ID = "value.output.logits"
SHA_RECORD_RE = re.compile(r"^[0-9a-f]{64}(?:\s+.+)?$")


def layer_value_id(layer: int) -> str:
    return f"value.layer.{layer}.output"


def validate_metrics(
    value: Any,
    row: dict[str, Any],
    *,
    shape: list[int],
    actual_dtype: str,
    label: str,
) -> dict[str, float]:
    base.require(isinstance(value, dict), f"{label} must be an object")
    base.require(value.get("shape") == shape, f"{label}.shape differs")
    base.require(value.get("element_count") == math.prod(shape),
                 f"{label}.element_count differs")
    base.require(value.get("actual_logical_dtype") == actual_dtype,
                 f"{label}.actual_logical_dtype differs")
    base.require(value.get("oracle_precision") == row["oracle_precision"],
                 f"{label}.oracle_precision differs")
    for name in (
        "actual_nan_count",
        "actual_inf_count",
        "expected_nan_count",
        "expected_inf_count",
    ):
        base.require(value.get(name) == 0, f"{label}.{name} != 0")
    for name in ("actual_f32_sha256", "expected_f32_sha256"):
        digest = value.get(name)
        base.require(
            isinstance(digest, str) and base.SHA256_RE.fullmatch(digest) is not None,
            f"{label}.{name} is not SHA256",
        )
    base.require(value.get("max_relative_error_denominator_floor") == 1.0e-12,
                 f"{label} relative-error denominator floor differs")
    measured = {
        name: base.finite_number(value.get(name), f"{label}.{name}")
        for name in ("max_abs", "max_relative_error", "relative_l2", "cosine")
    }
    bounds = row["bounds"]
    base.require(measured["max_abs"] <= bounds["max_abs_max"],
                 f"{label}.max_abs exceeds {bounds['max_abs_max']}")
    base.require(measured["relative_l2"] <= bounds["relative_l2_max"],
                 f"{label}.relative_l2 exceeds {bounds['relative_l2_max']}")
    base.require(measured["cosine"] >= bounds["cosine_min"],
                 f"{label}.cosine is below {bounds['cosine_min']}")
    return measured


def validate_diagnostic_metrics(
    value: Any,
    *,
    shape: list[int],
    label: str,
) -> dict[str, float]:
    base.require(isinstance(value, dict), f"{label} must be an object")
    base.require(value.get("shape") == shape and value.get("element_count") == math.prod(shape),
                 f"{label} shape/elements differ")
    for name in (
        "actual_nan_count",
        "actual_inf_count",
        "expected_nan_count",
        "expected_inf_count",
    ):
        base.require(value.get(name) == 0, f"{label}.{name} != 0")
    result = {
        name: base.finite_number(value.get(name), f"{label}.{name}")
        for name in ("max_abs", "max_relative_error", "relative_l2", "cosine")
    }
    base.require(result["max_abs"] >= 0.0 and result["relative_l2"] >= 0.0,
                 f"{label} has negative error")
    base.require(-1.0 <= result["cosine"] <= 1.0, f"{label}.cosine is invalid")
    return result


def validate_top_tokens(value: Any, label: str) -> dict[str, Any]:
    value = base.exact_fields(
        value,
        frozenset(
            {
                "argmax_token_id",
                "runner_up_token_id",
                "top_logit",
                "runner_up_logit",
                "top2_margin",
            }
        ),
        label,
    )
    for name in ("argmax_token_id", "runner_up_token_id"):
        token = value[name]
        base.require(isinstance(token, int) and not isinstance(token, bool)
                     and 0 <= token < VOCABULARY_SIZE,
                     f"{label}.{name} is invalid")
    top = base.finite_number(value["top_logit"], f"{label}.top_logit")
    runner_up = base.finite_number(value["runner_up_logit"],
                                   f"{label}.runner_up_logit")
    margin = base.finite_number(value["top2_margin"], f"{label}.top2_margin")
    base.require(value["argmax_token_id"] != value["runner_up_token_id"],
                 f"{label} top tokens are duplicated")
    base.require(top > runner_up and margin > 0.0,
                 f"{label} top-token ordering is invalid")
    base.require(math.isclose(top - runner_up, margin, rel_tol=1.0e-6, abs_tol=1.0e-6),
                 f"{label} top-2 margin differs")
    return {**value}


def validate_fixture(
    fixture: Any,
    full_model_row: dict[str, Any],
    logits_row: dict[str, Any],
) -> None:
    fixture = base.exact_fields(
        fixture,
        frozenset(
            {
                "fixture_id",
                "token_ids",
                "token_sequence_sha256",
                "tokens",
                "hidden_size",
                "layers",
                "full_attention_interval",
                "vocabulary_size",
                "quantized_embedding_is_model_input",
                "tied_embedding_lm_head",
            }
        ),
        "report.fixture",
    )
    token_ids = fixture["token_ids"]
    base.require(isinstance(token_ids, list) and len(token_ids) == TOKENS,
                 "fixture token IDs are incomplete")
    base.require(
        all(isinstance(value, int) and not isinstance(value, bool)
            and 0 <= value < 2**32 for value in token_ids),
        "fixture token IDs contain invalid values",
    )
    token_sha = hashlib.sha256(
        b"".join(int(value).to_bytes(4, "little") for value in token_ids)
    ).hexdigest()
    base.require(token_sha == fixture["token_sequence_sha256"] == EXPECTED_TOKEN_SHA256,
                 "fixture token sequence differs")
    base.require(
        fixture["tokens"] == TOKENS
        and fixture["hidden_size"] == HIDDEN_SIZE
        and fixture["layers"] == LAYERS
        and fixture["full_attention_interval"] == 4
        and fixture["vocabulary_size"] == VOCABULARY_SIZE
        and fixture["quantized_embedding_is_model_input"] is True
        and fixture["tied_embedding_lm_head"] is True,
        "fixture model dimensions/semantics differ",
    )
    full_domain = {
        "fixture_id": fixture["fixture_id"],
        "dimensions": {"hidden_size": HIDDEN_SIZE, "layers": LAYERS, "tokens": TOKENS},
        "semantics": {
            "final_norm": "rms_norm",
            "full_attention_interval": 4,
            "layer_pattern": "three_linear_then_one_full",
            "model_sha256": EXPECTED_MODEL_SHA256,
            "prompt_token_sequence_sha256": EXPECTED_TOKEN_SHA256,
            "quantized_embedding_is_model_input": True,
        },
    }
    base.require(full_domain == full_model_row["shape_domain"],
                 "full-model fixture differs from catalog row")
    logits_domain = {
        "fixture_id": "qwen35-4b.gguf-q4-k-m.full-vocab-logits.tokens-24",
        "dimensions": {"vocabulary_size": VOCABULARY_SIZE},
        "semantics": {
            "argmax_token_id": 57590,
            "last_token_index": 23,
            "model_sha256": EXPECTED_MODEL_SHA256,
            "prompt_token_sequence_sha256": EXPECTED_TOKEN_SHA256,
            "tied_embedding_lm_head": True,
        },
    }
    base.require(logits_domain == logits_row["shape_domain"],
                 "logits fixture differs from catalog row")


def validate_actual(actual: Any) -> tuple[dict[str, Path], dict[str, Any]]:
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
    artifact_root = Path(str(actual["artifact_root"])).resolve()
    capture_root = artifact_root / "capture"
    base.require(capture_root.is_dir(), "actual capture root is unavailable")
    for file_key, sha_key in (("plan_file", "plan_sha256"), ("wave_file", "wave_sha256")):
        path = base.safe_child(capture_root, actual[file_key], f"actual.{file_key}")
        base.require(base.sha256_file(path) == actual[sha_key],
                     f"actual {file_key} SHA mismatch")
    base.require(actual["tracked_diff_sha256"] == base.EMPTY_SHA256,
                 "actual capture contains tracked source changes")
    binary_record = actual["binary_sha256_record"]
    base.require(isinstance(binary_record, str)
                 and SHA_RECORD_RE.fullmatch(binary_record) is not None,
                 "actual binary SHA record is invalid")
    status = actual["git_status"]
    base.require(isinstance(status, str)
                 and all(line.startswith("?? ") for line in status.splitlines()),
                 "actual capture Git status contains tracked changes")
    base.require_git_commit(actual["git_sha"], "actual capture commit")

    expected_shapes = {EMBEDDING_VALUE_ID: FULL_HIDDEN_SHAPE}
    expected_shapes.update(
        {layer_value_id(layer): FULL_HIDDEN_SHAPE for layer in range(LAYERS)}
    )
    expected_shapes[FINAL_HIDDEN_VALUE_ID] = FULL_HIDDEN_SHAPE
    expected_shapes[LOGITS_VALUE_ID] = LOGITS_SHAPE
    checkpoints = base.exact_fields(
        actual["checkpoints"], frozenset(expected_shapes), "actual.checkpoints"
    )
    paths: dict[str, Path] = {}
    for value_id, shape in expected_shapes.items():
        checkpoint = base.exact_fields(
            checkpoints[value_id],
            frozenset({"logical_dtype", "logical_shape", "raw_file", "raw_sha256"}),
            f"actual.checkpoints.{value_id}",
        )
        base.require(checkpoint["logical_dtype"] == "fp16"
                     and checkpoint["logical_shape"] == shape,
                     f"{value_id} dtype/shape differs")
        raw = base.safe_child(capture_root, checkpoint["raw_file"], f"{value_id}.raw_file")
        base.require(raw.stat().st_size == math.prod(shape) * 2,
                     f"{value_id} byte count differs")
        base.require(base.sha256_file(raw) == checkpoint["raw_sha256"],
                     f"{value_id} raw SHA mismatch")
        paths[value_id] = raw
    return paths, checkpoints


def validate_reference_tensor(
    artifact_dir: Path,
    value: Any,
    *,
    shape: list[int],
    label: str,
    extra_fields: frozenset[str] = frozenset(),
) -> tuple[Path, dict[str, Any]]:
    value = base.exact_fields(
        value,
        frozenset({"raw_file", "raw_sha256", "logical_dtype", "logical_shape", "metrics"})
        | extra_fields,
        label,
    )
    base.require(value["logical_dtype"] == "fp32" and value["logical_shape"] == shape,
                 f"{label} dtype/shape differs")
    raw = base.safe_child(artifact_dir, value["raw_file"], f"{label}.raw_file")
    base.require(raw.stat().st_size == math.prod(shape) * 4,
                 f"{label} byte count differs")
    base.require(base.sha256_file(raw) == value["raw_sha256"], f"{label} SHA mismatch")
    base.require(value["metrics"].get("expected_f32_sha256") == value["raw_sha256"],
                 f"{label} metric/reference SHA differs")
    return raw, value


def validate_llama_cross_validation(value: Any) -> dict[str, Any]:
    value = base.exact_fields(
        value,
        frozenset(
            {
                "role",
                "provenance",
                "embedding_metrics",
                "layer_metrics",
                "final_hidden_metrics",
                "full_vocabulary_logits_metrics",
                "top_tokens",
            }
        ),
        "llama_cpp_cross_validation",
    )
    base.require(value["role"] == "same-GGUF quantized implementation cross-validation only",
                 "llama.cpp role is invalid")
    provenance = value["provenance"]
    base.require(isinstance(provenance, dict), "llama.cpp provenance must be an object")
    base.require(provenance.get("extractor_source_commit") == EXPECTED_EXTRACTOR_SOURCE_COMMIT
                 and provenance.get("extractor_source_sha256") == EXPECTED_EXTRACTOR_SOURCE_SHA256,
                 "llama extractor source provenance differs")
    base.require(isinstance(provenance.get("extractor_binary_sha256"), str)
                 and base.SHA256_RE.fullmatch(provenance["extractor_binary_sha256"]) is not None,
                 "llama extractor binary SHA is invalid")
    artifact_root = Path(str(provenance.get("artifact_root", ""))).resolve()
    manifest = artifact_root / "checkpoints/checkpoint-manifest.json"
    logits = artifact_root / "logits.f32"
    base.require(manifest.is_file() and logits.is_file(), "llama artifact files are unavailable")
    base.require(base.sha256_file(manifest) == provenance.get("manifest_sha256"),
                 "llama manifest SHA mismatch")
    base.require(base.sha256_file(logits) == provenance.get("logits_sha256"),
                 "llama logits SHA mismatch")
    embedding = validate_diagnostic_metrics(
        value["embedding_metrics"], shape=FULL_HIDDEN_SHAPE, label="llama.embedding"
    )
    layers = value["layer_metrics"]
    base.require(isinstance(layers, list) and len(layers) == LAYERS,
                 "llama layer metrics are incomplete")
    for layer, record in enumerate(layers):
        base.require(isinstance(record, dict)
                     and record.get("layer_index") == layer,
                     f"llama layer metric {layer} differs")
        shape = [1, HIDDEN_SIZE] if layer == 31 else FULL_HIDDEN_SHAPE
        validate_diagnostic_metrics(
            record.get("metrics"), shape=shape, label=f"llama.layer[{layer}]"
        )
    final_hidden = validate_diagnostic_metrics(
        value["final_hidden_metrics"], shape=[1, HIDDEN_SIZE], label="llama.final_hidden"
    )
    logits_metrics = validate_diagnostic_metrics(
        value["full_vocabulary_logits_metrics"], shape=LOGITS_SHAPE, label="llama.logits"
    )
    top = validate_top_tokens(value["top_tokens"], "llama.top_tokens")
    base.require(top["argmax_token_id"] == 57590,
                 "llama argmax differs from the reviewed fixture")
    thresholds_met = (
        final_hidden["cosine"] >= 0.999
        and final_hidden["relative_l2"] <= 0.01
        and logits_metrics["cosine"] >= 0.999
        and logits_metrics["relative_l2"] <= 0.01
    )
    return {
        "role": value["role"],
        "embedding": embedding,
        "final_hidden": final_hidden,
        "full_vocabulary_logits": logits_metrics,
        "top_tokens": top,
        "same_gguf_goal_thresholds_met": thresholds_met,
        "classification": "cross_validation_only" if thresholds_met else "diagnostic_miss",
    }


def validate_report(
    artifact_dir: Path,
    report: Any,
    full_model_row: dict[str, Any],
    logits_row: dict[str, Any],
) -> dict[str, Any]:
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
                "llama_cpp_cross_validation",
                "invocation",
            }
        ),
        "report",
    )
    base.forbid_artifact_tolerances(report)
    base.require(report["schema_version"] == 1 and report["status"] == "measured",
                 "report is not measured schema v1")
    oracle = report["oracle"]
    base.require(isinstance(oracle, dict)
                 and oracle.get("identity") == full_model_row["oracle_identity"]
                 == logits_row["oracle_identity"],
                 "oracle identity differs from catalog rows")
    source = oracle.get("ferrum_source")
    base.require(isinstance(source, dict) and source.get("tracked_dirty") is False,
                 "oracle source worktree was dirty")
    base.require(source.get("git_sha") == full_model_row["source_commit"]
                 == logits_row["source_commit"],
                 "oracle source commit differs")
    source_payload = base.git_bytes(full_model_row["source_commit"],
                                    full_model_row["basis"]["source_path"])
    base.require(hashlib.sha256(source_payload).hexdigest() == EXPECTED_SOURCE_SHA256
                 and oracle.get("source_sha256") == EXPECTED_SOURCE_SHA256,
                 "model oracle source hash differs")
    for path_key, sha_key, expected_path, expected_sha in (
        (
            "linear_common_source_path",
            "linear_common_source_sha256",
            "scripts/release/qwen35_gguf_linear_attention_reference.py",
            EXPECTED_LINEAR_COMMON_SHA256,
        ),
        (
            "full_attention_common_source_path",
            "full_attention_common_source_sha256",
            "scripts/release/qwen35_gguf_full_attention_reference.py",
            EXPECTED_FULL_COMMON_SHA256,
        ),
    ):
        base.require(oracle.get(path_key) == expected_path, f"oracle {path_key} differs")
        payload = base.git_bytes(full_model_row["source_commit"], expected_path)
        base.require(hashlib.sha256(payload).hexdigest() == expected_sha
                     and oracle.get(sha_key) == expected_sha,
                     f"oracle {sha_key} differs")
    llama_source = oracle.get("llama_cpp_gguf_py_source")
    base.require(isinstance(llama_source, dict)
                 and llama_source.get("tracked_dirty") is False,
                 "gguf-py source worktree was dirty")
    base.require_git_commit(source["git_sha"], "oracle source commit")

    shared.validate_model(artifact_dir, report["model"])
    validate_fixture(report["fixture"], full_model_row, logits_row)
    actual_paths, actual_checkpoints = validate_actual(report["actual"])
    reference = base.exact_fields(
        report["reference"],
        frozenset(
            {
                "embedding",
                "layers",
                "final_hidden",
                "full_vocabulary_logits",
                "isolated_ferrum_final_hidden_head",
                "actual_top_tokens",
            }
        ),
        "report.reference",
    )
    _, embedding = validate_reference_tensor(
        artifact_dir, reference["embedding"], shape=FULL_HIDDEN_SHAPE,
        label="reference.embedding"
    )
    validate_metrics(
        embedding["metrics"], full_model_row,
        shape=FULL_HIDDEN_SHAPE, actual_dtype="fp16", label="embedding.metrics"
    )
    base.require(
        embedding["metrics"]["actual_f32_sha256"]
        == base.f16_as_f32_sha256(actual_paths[EMBEDDING_VALUE_ID], FULL_HIDDEN_ELEMENTS),
        "embedding metric/actual SHA differs",
    )

    layers = reference["layers"]
    base.require(isinstance(layers, list) and len(layers) == LAYERS,
                 "reference layers are incomplete")
    worst_layer: dict[str, Any] | None = None
    for layer, record in enumerate(layers):
        record = base.exact_fields(
            record,
            frozenset(
                {"layer_index", "layer_kind", "raw_file", "raw_sha256", "actual_raw_sha256", "metrics"}
            ),
            f"reference.layers[{layer}]",
        )
        expected_kind = "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
        base.require(record["layer_index"] == layer and record["layer_kind"] == expected_kind,
                     f"reference layer {layer} identity differs")
        raw = base.safe_child(artifact_dir, record["raw_file"], f"reference.layer[{layer}].raw")
        base.require(raw.stat().st_size == FULL_HIDDEN_ELEMENTS * 4
                     and base.sha256_file(raw) == record["raw_sha256"],
                     f"reference layer {layer} raw tensor differs")
        base.require(record["actual_raw_sha256"]
                     == actual_checkpoints[layer_value_id(layer)]["raw_sha256"],
                     f"reference layer {layer} actual SHA differs")
        metrics = validate_metrics(
            record["metrics"], full_model_row,
            shape=FULL_HIDDEN_SHAPE, actual_dtype="fp16",
            label=f"reference.layer[{layer}].metrics",
        )
        base.require(record["metrics"]["expected_f32_sha256"] == record["raw_sha256"],
                     f"reference layer {layer} metric SHA differs")
        base.require(
            record["metrics"]["actual_f32_sha256"]
            == base.f16_as_f32_sha256(actual_paths[layer_value_id(layer)], FULL_HIDDEN_ELEMENTS),
            f"reference layer {layer} actual metric SHA differs",
        )
        if worst_layer is None or metrics["relative_l2"] > worst_layer["relative_l2"]:
            worst_layer = {"layer_index": layer, "layer_kind": expected_kind, **metrics}

    final_path, final_hidden = validate_reference_tensor(
        artifact_dir, reference["final_hidden"], shape=FULL_HIDDEN_SHAPE,
        label="reference.final_hidden"
    )
    final_metrics = validate_metrics(
        final_hidden["metrics"], full_model_row,
        shape=FULL_HIDDEN_SHAPE, actual_dtype="fp16", label="final_hidden.metrics"
    )
    base.require(
        final_hidden["metrics"]["actual_f32_sha256"]
        == base.f16_as_f32_sha256(actual_paths[FINAL_HIDDEN_VALUE_ID], FULL_HIDDEN_ELEMENTS),
        "final-hidden metric/actual SHA differs",
    )
    logits_path, logits = validate_reference_tensor(
        artifact_dir, reference["full_vocabulary_logits"], shape=LOGITS_SHAPE,
        label="reference.full_vocabulary_logits", extra_fields=frozenset({"top_tokens"})
    )
    logits_metrics = validate_metrics(
        logits["metrics"], logits_row,
        shape=LOGITS_SHAPE, actual_dtype="fp16", label="logits.metrics"
    )
    base.require(
        logits["metrics"]["actual_f32_sha256"]
        == base.f16_as_f32_sha256(actual_paths[LOGITS_VALUE_ID], VOCABULARY_SIZE),
        "logits metric/actual SHA differs",
    )
    logits_top = validate_top_tokens(logits["top_tokens"], "reference.logits.top_tokens")
    actual_top = validate_top_tokens(reference["actual_top_tokens"], "reference.actual_top_tokens")
    base.require(logits_top["argmax_token_id"] == actual_top["argmax_token_id"] == 57590,
                 "reference/actual argmax differs")
    base.require(logits_top["runner_up_token_id"] == actual_top["runner_up_token_id"],
                 "reference/actual runner-up token differs")

    isolated_path, isolated = validate_reference_tensor(
        artifact_dir, reference["isolated_ferrum_final_hidden_head"], shape=LOGITS_SHAPE,
        label="reference.isolated_head", extra_fields=frozenset({"top_tokens"})
    )
    isolated_metrics = validate_metrics(
        isolated["metrics"], logits_row,
        shape=LOGITS_SHAPE, actual_dtype="fp16", label="isolated_head.metrics"
    )
    isolated_top = validate_top_tokens(isolated["top_tokens"], "reference.isolated_head.top_tokens")
    base.require(isolated_top["argmax_token_id"] == 57590,
                 "isolated head argmax differs")
    llama = validate_llama_cross_validation(report["llama_cpp_cross_validation"])
    return {
        "full_model_metrics": final_metrics,
        "full_vocabulary_logits_metrics": logits_metrics,
        "isolated_head_metrics": isolated_metrics,
        "worst_layer_metrics": worst_layer,
        "argmax_token_id": 57590,
        "reference_top2_margin": logits_top["top2_margin"],
        "actual_top2_margin": actual_top["top2_margin"],
        "final_hidden_raw_sha256": base.sha256_file(final_path),
        "full_vocabulary_logits_raw_sha256": base.sha256_file(logits_path),
        "isolated_head_raw_sha256": base.sha256_file(isolated_path),
        "llama_cpp_cross_validation": llama,
    }


def self_test() -> None:
    row = {
        "oracle_precision": "fp32",
        "bounds": {"cosine_min": 0.999, "relative_l2_max": 0.01, "max_abs_max": 0.25},
    }
    metrics = {
        "shape": FULL_HIDDEN_SHAPE,
        "element_count": FULL_HIDDEN_ELEMENTS,
        "actual_logical_dtype": "fp16",
        "oracle_precision": "fp32",
        "actual_nan_count": 0,
        "actual_inf_count": 0,
        "expected_nan_count": 0,
        "expected_inf_count": 0,
        "actual_f32_sha256": "a" * 64,
        "expected_f32_sha256": "b" * 64,
        "max_abs": 0.083,
        "max_relative_error": 80.0,
        "max_relative_error_denominator_floor": 1.0e-12,
        "relative_l2": 0.0018,
        "cosine": 0.999998,
    }
    validate_metrics(
        metrics, row, shape=FULL_HIDDEN_SHAPE,
        actual_dtype="fp16", label="selftest"
    )
    rejected = dict(metrics)
    rejected["cosine"] = 0.998
    try:
        validate_metrics(
            rejected, row, shape=FULL_HIDDEN_SHAPE,
            actual_dtype="fp16", label="selftest"
        )
    except base.GateError as error:
        base.require("cosine" in str(error), "wrong model metric rejection")
    else:
        raise base.GateError("out-of-bound model metric unexpectedly passed")
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
        base.require(args.artifact_dir is not None and args.out is not None,
                     "artifact_dir and --out are required")
        artifact_dir = Path(args.artifact_dir).expanduser().resolve()
        out_dir = Path(args.out).expanduser().resolve()
        base.require(artifact_dir.is_dir(), "reference artifact directory is unavailable")
        base.require(not out_dir.exists() or out_dir.is_dir(), "--out is not a directory")
        base.require(not out_dir.exists() or not any(out_dir.iterdir()),
                     "--out directory is not empty")
        catalog, provenance = tolerances.load_catalog_from_git(
            args.git_revision, args.expected_catalog_blob_sha
        )
        summary = tolerances.validate_catalog_document(catalog, require_complete=True)
        tolerances.validate_catalog_provenance(catalog, provenance["commit"])
        compared = tolerances.validate_no_widening_from_revision(catalog, provenance["commit"])
        base.require(summary["coverage_status"] == "complete"
                     and summary["missing_required_coverage"] == [],
                     "catalog numerical coverage is not complete")
        rows = {row["tolerance_id"]: row for row in catalog["rows"]}
        base.require(FULL_MODEL_TOLERANCE_ID in rows and LOGITS_TOLERANCE_ID in rows,
                     "catalog model rows are missing")
        full_model_row = rows[FULL_MODEL_TOLERANCE_ID]
        logits_row = rows[LOGITS_TOLERANCE_ID]
        base.require(full_model_row["coverage_markers"] == ["checkpoint.full_model"],
                     "full-model marker differs")
        base.require(logits_row["coverage_markers"] == ["checkpoint.full_vocab_logits"],
                     "logits marker differs")
        report_path = artifact_dir / "report.json"
        validation = validate_report(
            artifact_dir, base.load_json(report_path), full_model_row, logits_row
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        base.write_json(
            out_dir / "qwen35-model-numerics.gate.json",
            {
                "schema_version": 1,
                "status": "pass",
                "scope": "independent Qwen3.5 full-model and full-vocabulary numerical reference",
                "reference_artifact": str(artifact_dir),
                "reference_report_sha256": base.sha256_file(report_path),
                "catalog_commit": provenance["commit"],
                "catalog_git_blob_sha": provenance["git_blob_sha"],
                "catalog_coverage_status": summary["coverage_status"],
                "compared_catalog_commits": compared,
                "tolerances": {
                    FULL_MODEL_TOLERANCE_ID: full_model_row["row_fingerprint"],
                    LOGITS_TOLERANCE_ID: logits_row["row_fingerprint"],
                },
                **validation,
            },
        )
    except (base.GateError, tolerances.CatalogError) as error:
        print(f"RUNTIME VNEXT QWEN35 MODEL NUMERICS FAIL: {error}", file=sys.stderr)
        return 1
    print(f"{PASS_PREFIX}: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
