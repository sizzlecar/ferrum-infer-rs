#!/usr/bin/env python3
"""Validate G08A Ferrum/CPU/llama.cpp logits on one canonical token history."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from array import array
from pathlib import Path
from typing import Any, BinaryIO

import runtime_vnext_checkpoint_artifact as checkpoint_artifact


REPO_ROOT = Path(__file__).resolve().parents[2]
PASS_PREFIX = "FERRUM RUNTIME VNEXT G08A SAME HISTORY PASS"
FAIL_PREFIX = "FERRUM RUNTIME VNEXT G08A SAME HISTORY FAIL"
SELFTEST_PASS = "FERRUM RUNTIME VNEXT G08A SAME HISTORY SELFTEST PASS"
SCHEMA_VERSION = 1
MODEL_ID = "unsloth/Qwen3.5-4B-GGUF"
MODEL_SHA256 = "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
MODEL_REVISION = "e87f176479d0855a907a41277aca2f8ee7a09523"
DECISION_COUNT = 64
VOCABULARY_SIZE = 248_320
ROW_BYTES = VOCABULARY_SIZE * 4
ORACLE_AMBIGUITY_MARGIN = 1.0e-3
TOLERANCE_ID = (
    "runtime-vnext.metal.qwen35-4b.same-history-logits.v1."
    "full-vocab-logits.fp32.gguf-q4-k-m.decisions-64"
)
SHA256_RE_LENGTH = 64

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
        "tolerance_id",
        "oracle_ambiguity_margin",
        "prompt_id",
        "prompt_sha256",
        "prompt_token_ids_sha256",
        "teacher_token_ids_sha256",
        "teacher_token_count",
        "ferrum_capture_dir",
        "llama_capture_dir",
        "oracle_dir",
        "comparison",
        "exception_count",
        "waiver_count",
    }
)
COMPARISON_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "decision_count",
        "vocabulary_size",
        "raw_sets",
        "metrics",
        "decisions",
        "summary",
    }
)
RAW_SET_FIELDS = frozenset({"decision_count", "row_bytes", "ordered_sha256"})
METRIC_FIELDS = frozenset({"minimum_cosine", "maximum_relative_l2", "maximum_abs"})
DECISION_FIELDS = frozenset(
    {
        "index",
        "oracle_margin",
        "oracle_top2_token_ids",
        "oracle_top2_logits",
        "ferrum_argmax_token_id",
        "llama_argmax_token_id",
        "classification",
        "status",
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


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def reject_constant(value: str) -> None:
    raise GateError(f"non-finite JSON constant is forbidden: {value}")


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"cannot read {label} {path}: {error}") from error
    require(isinstance(value, dict), f"{label} must be an object")
    return value


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


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{label} must be an integer >= {minimum}",
    )
    return value


def sha256_text(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == SHA256_RE_LENGTH
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA256",
    )
    return value


def git_sha_text(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase full Git SHA",
    )
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def ordered_raw_sha256(paths: list[Path]) -> str:
    return canonical_sha256(
        [
            {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in paths
        ]
    )


def token_ids_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        require(0 <= token_id < 2**32, "token id must fit u32")
        digest.update(token_id.to_bytes(4, "little"))
    return digest.hexdigest()


def load_f32_row(handle: BinaryIO, count: int, label: str) -> array:
    values = array("f")
    try:
        values.fromfile(handle, count)
    except (EOFError, ValueError) as error:
        raise GateError(f"{label} is truncated") from error
    require(len(values) == count, f"{label} element count differs")
    if sys.byteorder != "little":
        values.byteswap()
    require(all(math.isfinite(value) for value in values), f"{label} contains NaN or Inf")
    return values


def argmax_and_top2(values: array, label: str) -> tuple[int, list[int], list[float]]:
    require(len(values) >= 2, f"{label} must contain at least two values")
    first_value = max(values)
    first_index = values.index(first_value)
    values[first_index] = -math.inf
    try:
        second_value = max(values)
        second_index = values.index(second_value)
    finally:
        values[first_index] = first_value
    return first_index, [first_index, second_index], [float(first_value), float(second_value)]


def validate_metric(value: Any, label: str) -> dict[str, float]:
    metric = exact_object(value, METRIC_FIELDS, label)
    return {field: finite(metric[field], f"{label}.{field}") for field in METRIC_FIELDS}


def validate_tolerance_row(value: dict[str, Any]) -> dict[str, float]:
    require(value.get("tolerance_id") == TOLERANCE_ID, "same-history tolerance ID differs")
    bounds = value.get("bounds")
    require(isinstance(bounds, dict), "same-history tolerance bounds are missing")
    required = {"cosine_min", "relative_l2_max", "max_abs_max"}
    require(set(bounds) == required, "same-history tolerance bound fields differ")
    return {field: finite(bounds[field], f"tolerance.{field}") for field in required}


def validate_metric_bounds(metric: dict[str, float], bounds: dict[str, float]) -> None:
    require(metric["minimum_cosine"] >= bounds["cosine_min"], "same-history cosine is below tolerance")
    require(
        metric["maximum_relative_l2"] <= bounds["relative_l2_max"],
        "same-history relative L2 exceeds tolerance",
    )
    require(metric["maximum_abs"] <= bounds["max_abs_max"], "same-history max abs exceeds tolerance")


def decision_classification(margin: float) -> str:
    require(margin >= 0.0 and math.isfinite(margin), "oracle margin must be finite and non-negative")
    return "ambiguous" if margin < ORACLE_AMBIGUITY_MARGIN else "robust"


def enforce_decision_policy(
    *,
    index: int,
    classification: str,
    ferrum_argmax: int,
    oracle_argmax: int,
    oracle_top2: list[int],
) -> None:
    require(classification in {"robust", "ambiguous"}, f"decision[{index}] classification is invalid")
    if classification == "robust":
        require(
            ferrum_argmax == oracle_argmax,
            f"robust decision[{index}] Ferrum argmax differs from CPU oracle",
        )
    else:
        require(
            ferrum_argmax in oracle_top2,
            f"ambiguous decision[{index}] Ferrum argmax is outside oracle top2",
        )


def validate_llama_manifest(
    capture_dir: Path,
    *,
    prompt_tokens: list[int],
    teacher_tokens: list[int],
) -> tuple[list[Path], dict[str, Any]]:
    manifest = read_json(capture_dir / "manifest.json", "llama manifest")
    require(
        manifest.get("schema") == "ferrum.llama-teacher-logits-dump.v1"
        and manifest.get("schema_version") == 1
        and manifest.get("status") == "pass",
        "llama manifest identity is invalid",
    )
    backend = manifest.get("backend")
    require(
        isinstance(backend, dict)
        and backend.get("name") == "metal"
        and backend.get("n_gpu_layers") == -1,
        "llama manifest did not use full Metal offload",
    )
    expected_input = prompt_tokens + teacher_tokens[:-1]
    input_record = manifest.get("input")
    require(
        isinstance(input_record, dict)
        and input_record.get("architecture") == "qwen35"
        and input_record.get("token_count") == len(expected_input)
        and input_record.get("token_ids") == expected_input,
        "llama canonical input history differs",
    )
    require(
        manifest.get("prefill") == {"token_count": len(prompt_tokens), "decode_calls": 1},
        "llama prefill contract differs",
    )
    require(
        manifest.get("decode")
        == {"teacher_token_count": len(teacher_tokens) - 1, "decode_calls": len(teacher_tokens) - 1},
        "llama decode contract differs",
    )
    decision = manifest.get("decision")
    require(
        isinstance(decision, dict)
        and decision.get("count") == len(teacher_tokens)
        and isinstance(decision.get("records"), list),
        "llama decision contract differs",
    )
    records = decision["records"]
    require(len(records) == len(teacher_tokens), "llama decision denominator differs")
    paths: list[Path] = []
    for index, record in enumerate(records):
        require(isinstance(record, dict), f"llama decision[{index}] must be an object")
        expected_name = f"decision-{index:04d}.f32"
        require(
            record.get("index") == index
            and record.get("file") == expected_name
            and record.get("bytes") == ROW_BYTES,
            f"llama decision[{index}] identity differs",
        )
        expected_context = len(prompt_tokens) + index
        require(
            record.get("context_token_count") == expected_context,
            f"llama decision[{index}] context length differs",
        )
        path = capture_dir / expected_name
        require(path.is_file() and not path.is_symlink(), f"llama decision[{index}] raw file is missing")
        require(path.stat().st_size == ROW_BYTES, f"llama decision[{index}] raw size differs")
        paths.append(path)
    require(manifest.get("vocab") == {"size": VOCABULARY_SIZE}, "llama vocabulary differs")
    require(manifest.get("dtype") == "f32", "llama logits dtype differs")
    execution = manifest.get("execution")
    require(
        isinstance(execution, dict)
        and execution.get("parallel_sequences") == 1
        and execution.get("n_seq_max") == 1
        and 1 <= integer(execution.get("worker_threads"), "llama worker_threads", minimum=1) <= 4,
        "llama execution bounds differ",
    )
    return paths, manifest


def validate_oracle_report(
    oracle_dir: Path,
    *,
    source_git_sha: str,
    prompt_token_sha256: str,
    teacher_token_sha256: str,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    report = read_json(oracle_dir / "report.json", "CPU oracle report")
    require(report.get("schema_version") == 1 and report.get("status") == "pass", "CPU oracle report is not PASS")
    model = report.get("model")
    require(
        isinstance(model, dict)
        and model.get("sha256") == MODEL_SHA256
        and model.get("format") == "GGUF Q4_K_M",
        "CPU oracle model identity differs",
    )
    oracle = report.get("oracle")
    require(
        isinstance(oracle, dict)
        and oracle.get("identity") == "cpu.fp32.python.qwen35_gguf_teacher_logits_reference"
        and oracle.get("precision") == "fp32",
        "CPU oracle identity differs",
    )
    ferrum_source = oracle.get("ferrum_source")
    require(
        isinstance(ferrum_source, dict)
        and ferrum_source.get("git_sha") == source_git_sha
        and ferrum_source.get("tracked_dirty") is False,
        "CPU oracle Ferrum source identity is stale or dirty",
    )
    input_record = report.get("input")
    require(
        isinstance(input_record, dict)
        and input_record.get("prompt_token_ids_sha256") == prompt_token_sha256
        and input_record.get("teacher_token_ids_sha256") == teacher_token_sha256
        and input_record.get("teacher_token_count") == DECISION_COUNT,
        "CPU oracle canonical history differs",
    )
    output = report.get("output")
    require(isinstance(output, dict), "CPU oracle output is missing")
    require(
        output.get("logical_dtype") == "fp32"
        and output.get("logical_shape") == [DECISION_COUNT, VOCABULARY_SIZE]
        and output.get("raw_bytes") == DECISION_COUNT * ROW_BYTES
        and output.get("nan_count") == 0
        and output.get("inf_count") == 0,
        "CPU oracle output shape/dtype/finite contract differs",
    )
    raw_name = output.get("raw_file")
    require(raw_name == "decision-logits.f32", "CPU oracle raw filename differs")
    raw_path = oracle_dir / raw_name
    require(raw_path.is_file() and not raw_path.is_symlink(), "CPU oracle raw logits are missing")
    require(raw_path.stat().st_size == DECISION_COUNT * ROW_BYTES, "CPU oracle raw size differs")
    require(file_sha256(raw_path) == output.get("raw_sha256"), "CPU oracle raw SHA differs")
    decisions = output.get("decisions")
    require(isinstance(decisions, list) and len(decisions) == DECISION_COUNT, "CPU oracle decision denominator differs")
    return raw_path, decisions, report


def ferrum_paths_and_history(capture_dir: Path) -> tuple[list[Path], list[int], list[int], dict[str, Any]]:
    try:
        summary = checkpoint_artifact.validate_artifact(capture_dir, MODEL_ID, [])
    except checkpoint_artifact.ArtifactError as error:
        raise GateError(f"Ferrum checkpoint artifact rejected: {error}") from error
    teacher = summary.get("teacher_forcing")
    require(
        isinstance(teacher, dict)
        and teacher.get("mode") == "canonical-history"
        and teacher.get("token_count") == DECISION_COUNT,
        "Ferrum checkpoint teacher contract differs",
    )
    prompt_record = read_json(capture_dir / "teacher-prompt.json", "Ferrum teacher prompt")
    prompt_tokens = prompt_record.get("token_ids")
    require(isinstance(prompt_tokens, list) and prompt_tokens, "Ferrum prompt token ids are missing")
    decisions = [wave.get("teacher_forced_decision") for wave in summary["waves"]]
    require(all(isinstance(value, dict) for value in decisions), "Ferrum teacher decisions are missing")
    teacher_tokens = [int(value["token_id"]) for value in decisions]
    paths: list[Path] = []
    for index, wave in enumerate(summary["waves"]):
        outputs = wave.get("product_outputs")
        require(isinstance(outputs, list) and len(outputs) == 1, f"Ferrum decision[{index}] output differs")
        output = outputs[0]
        require(
            output.get("output_mode") == "full-logits"
            and output.get("element_type") == "f32"
            and output.get("element_count") == VOCABULARY_SIZE
            and output.get("raw_bytes") == ROW_BYTES,
            f"Ferrum decision[{index}] logits contract differs",
        )
        path = capture_dir / str(output["raw_file"])
        require(path.is_file() and path.stat().st_size == ROW_BYTES, f"Ferrum decision[{index}] raw file differs")
        paths.append(path)
    return paths, prompt_tokens, teacher_tokens, summary


def validate_comparison(
    comparison_path: Path,
    *,
    ferrum_paths: list[Path],
    llama_paths: list[Path],
    oracle_path: Path,
    oracle_decisions: list[dict[str, Any]],
    bounds: dict[str, float],
) -> dict[str, int]:
    comparison = exact_object(read_json(comparison_path, "same-history comparison"), COMPARISON_FIELDS, "comparison")
    require(
        comparison["schema_version"] == 1
        and comparison["status"] == "pass"
        and comparison["decision_count"] == DECISION_COUNT
        and comparison["vocabulary_size"] == VOCABULARY_SIZE,
        "same-history comparison identity differs",
    )
    raw_sets = exact_object(comparison["raw_sets"], frozenset({"ferrum", "llama", "oracle"}), "comparison.raw_sets")
    expected_paths = {"ferrum": ferrum_paths, "llama": llama_paths, "oracle": [oracle_path]}
    for source, paths in expected_paths.items():
        raw = exact_object(raw_sets[source], RAW_SET_FIELDS, f"comparison.raw_sets.{source}")
        require(raw["decision_count"] == DECISION_COUNT and raw["row_bytes"] == ROW_BYTES, f"{source} raw denominator differs")
        require(raw["ordered_sha256"] == ordered_raw_sha256(paths), f"{source} ordered raw SHA differs")
    metrics = exact_object(comparison["metrics"], frozenset({"ferrum_vs_oracle", "llama_vs_oracle", "ferrum_vs_llama"}), "comparison.metrics")
    ferrum_metric = validate_metric(metrics["ferrum_vs_oracle"], "ferrum_vs_oracle")
    validate_metric(metrics["llama_vs_oracle"], "llama_vs_oracle")
    validate_metric(metrics["ferrum_vs_llama"], "ferrum_vs_llama")
    validate_metric_bounds(ferrum_metric, bounds)
    rows = comparison["decisions"]
    require(isinstance(rows, list) and len(rows) == DECISION_COUNT, "comparison decision denominator differs")
    counts = {
        "robust_decision_count": 0,
        "ambiguous_decision_count": 0,
        "ferrum_oracle_exact_count": 0,
        "ambiguous_top2_accepted_count": 0,
        "llama_oracle_exact_count": 0,
        "external_flip_count": 0,
        "exception_count": 0,
        "waiver_count": 0,
    }
    with oracle_path.open("rb") as oracle_handle:
        for index, raw in enumerate(rows):
            row = exact_object(raw, DECISION_FIELDS, f"comparison.decisions[{index}]")
            require(row["index"] == index and row["status"] == "pass", f"decision[{index}] identity/status differs")
            with ferrum_paths[index].open("rb") as ferrum_handle:
                ferrum_values = load_f32_row(ferrum_handle, VOCABULARY_SIZE, f"Ferrum decision[{index}]")
                require(not ferrum_handle.read(1), f"Ferrum decision[{index}] has trailing bytes")
            with llama_paths[index].open("rb") as llama_handle:
                llama_values = load_f32_row(llama_handle, VOCABULARY_SIZE, f"llama decision[{index}]")
                require(not llama_handle.read(1), f"llama decision[{index}] has trailing bytes")
            oracle_values = load_f32_row(oracle_handle, VOCABULARY_SIZE, f"oracle decision[{index}]")
            ferrum_argmax, _, _ = argmax_and_top2(ferrum_values, f"Ferrum decision[{index}]")
            llama_argmax, _, _ = argmax_and_top2(llama_values, f"llama decision[{index}]")
            oracle_argmax, oracle_top2, oracle_top2_logits = argmax_and_top2(oracle_values, f"oracle decision[{index}]")
            oracle_record = oracle_decisions[index]
            require(isinstance(oracle_record, dict), f"CPU oracle decision[{index}] record differs")
            margin = oracle_top2_logits[0] - oracle_top2_logits[1]
            require(
                oracle_record.get("token_index") == index
                and oracle_record.get("top2_token_ids") == oracle_top2
                and all(
                    math.isclose(float(left), right, rel_tol=0.0, abs_tol=1.0e-6)
                    for left, right in zip(oracle_record.get("top2_logits", []), oracle_top2_logits, strict=True)
                )
                and math.isclose(float(oracle_record.get("margin")), margin, rel_tol=0.0, abs_tol=1.0e-6),
                f"CPU oracle decision[{index}] report differs from raw logits",
            )
            classification = decision_classification(margin)
            require(
                row["classification"] == classification
                and math.isclose(finite(row["oracle_margin"], f"decision[{index}].oracle_margin"), margin, rel_tol=0.0, abs_tol=1.0e-6)
                and row["oracle_top2_token_ids"] == oracle_top2
                and row["ferrum_argmax_token_id"] == ferrum_argmax
                and row["llama_argmax_token_id"] == llama_argmax,
                f"decision[{index}] comparison differs from raw logits",
            )
            reported_top2 = row["oracle_top2_logits"]
            require(
                isinstance(reported_top2, list)
                and len(reported_top2) == 2
                and all(
                    math.isclose(finite(left, f"decision[{index}].oracle_top2_logits"), right, rel_tol=0.0, abs_tol=1.0e-6)
                    for left, right in zip(reported_top2, oracle_top2_logits, strict=True)
                ),
                f"decision[{index}] oracle top2 logits differ",
            )
            if classification == "robust":
                counts["robust_decision_count"] += 1
            else:
                counts["ambiguous_decision_count"] += 1
                counts["ambiguous_top2_accepted_count"] += 1
            enforce_decision_policy(
                index=index,
                classification=classification,
                ferrum_argmax=ferrum_argmax,
                oracle_argmax=oracle_argmax,
                oracle_top2=oracle_top2,
            )
            counts["ferrum_oracle_exact_count"] += int(ferrum_argmax == oracle_argmax)
            counts["llama_oracle_exact_count"] += int(llama_argmax == oracle_argmax)
            counts["external_flip_count"] += int(llama_argmax != oracle_argmax)
        require(not oracle_handle.read(1), "CPU oracle raw logits have trailing bytes")
    summary = exact_object(comparison["summary"], SUMMARY_FIELDS, "comparison.summary")
    require(summary == counts, "comparison summary differs from recomputed decisions")
    return counts


def build_comparison(
    *,
    ferrum_capture_dir: Path,
    llama_capture_dir: Path,
    oracle_dir: Path,
    tolerance_row: dict[str, Any],
    out: Path,
) -> dict[str, Any]:
    try:
        import numpy as np  # type: ignore
    except ImportError as error:
        raise GateError("numpy is required; run comparison with `uv run --with numpy`") from error
    ferrum_paths, prompt_tokens, teacher_tokens, _ferrum_summary = ferrum_paths_and_history(
        ferrum_capture_dir
    )
    llama_paths, _llama_manifest = validate_llama_manifest(
        llama_capture_dir,
        prompt_tokens=prompt_tokens,
        teacher_tokens=teacher_tokens,
    )
    oracle_report = read_json(oracle_dir / "report.json", "CPU oracle report")
    oracle_source = oracle_report.get("oracle", {}).get("ferrum_source", {})
    require(isinstance(oracle_source, dict), "CPU oracle Ferrum source identity is missing")
    oracle_path, oracle_decisions, _oracle_report = validate_oracle_report(
        oracle_dir,
        source_git_sha=str(oracle_source.get("git_sha", "")),
        prompt_token_sha256=token_ids_sha256(prompt_tokens),
        teacher_token_sha256=token_ids_sha256(teacher_tokens),
    )
    bounds = validate_tolerance_row(tolerance_row)
    oracle = np.memmap(
        oracle_path,
        dtype="<f4",
        mode="r",
        shape=(DECISION_COUNT, VOCABULARY_SIZE),
    )
    minimum_cosines = {"ferrum_vs_oracle": 1.0, "llama_vs_oracle": 1.0, "ferrum_vs_llama": 1.0}
    maximum_relative_l2 = {key: 0.0 for key in minimum_cosines}
    maximum_abs = {key: 0.0 for key in minimum_cosines}
    decisions: list[dict[str, Any]] = []
    counts = {
        "robust_decision_count": 0,
        "ambiguous_decision_count": 0,
        "ferrum_oracle_exact_count": 0,
        "ambiguous_top2_accepted_count": 0,
        "llama_oracle_exact_count": 0,
        "external_flip_count": 0,
        "exception_count": 0,
        "waiver_count": 0,
    }
    for index in range(DECISION_COUNT):
        ferrum = np.fromfile(ferrum_paths[index], dtype="<f4")
        llama = np.fromfile(llama_paths[index], dtype="<f4")
        expected = np.asarray(oracle[index])
        require(
            ferrum.shape == llama.shape == expected.shape == (VOCABULARY_SIZE,),
            f"decision[{index}] raw shape differs",
        )
        require(
            bool(np.isfinite(ferrum).all())
            and bool(np.isfinite(llama).all())
            and bool(np.isfinite(expected).all()),
            f"decision[{index}] raw logits contain NaN or Inf",
        )
        pairs = {
            "ferrum_vs_oracle": (ferrum, expected),
            "llama_vs_oracle": (llama, expected),
            "ferrum_vs_llama": (ferrum, llama),
        }
        for label, (actual_raw, expected_raw) in pairs.items():
            actual = actual_raw.astype(np.float64)
            reference = expected_raw.astype(np.float64)
            actual_norm = float(np.linalg.norm(actual))
            reference_norm = float(np.linalg.norm(reference))
            require(actual_norm > 0.0 and reference_norm > 0.0, f"{label} has a zero norm")
            cosine = float(np.dot(actual, reference) / (actual_norm * reference_norm))
            relative_l2 = float(np.linalg.norm(actual - reference) / reference_norm)
            max_abs = float(np.max(np.abs(actual - reference)))
            minimum_cosines[label] = min(minimum_cosines[label], cosine)
            maximum_relative_l2[label] = max(maximum_relative_l2[label], relative_l2)
            maximum_abs[label] = max(maximum_abs[label], max_abs)
        ferrum_argmax = int(np.argmax(ferrum))
        llama_argmax = int(np.argmax(llama))
        oracle_order = np.argsort(expected)[-2:][::-1]
        oracle_top2 = [int(token) for token in oracle_order]
        oracle_top2_logits = [float(expected[token]) for token in oracle_order]
        oracle_argmax = oracle_top2[0]
        margin = oracle_top2_logits[0] - oracle_top2_logits[1]
        oracle_record = oracle_decisions[index]
        require(
            oracle_record.get("top2_token_ids") == oracle_top2
            and all(
                math.isclose(float(left), right, rel_tol=0.0, abs_tol=1.0e-6)
                for left, right in zip(oracle_record.get("top2_logits", []), oracle_top2_logits, strict=True)
            ),
            f"CPU oracle decision[{index}] report differs from raw logits",
        )
        classification = decision_classification(margin)
        enforce_decision_policy(
            index=index,
            classification=classification,
            ferrum_argmax=ferrum_argmax,
            oracle_argmax=oracle_argmax,
            oracle_top2=oracle_top2,
        )
        counts[f"{classification}_decision_count"] += 1
        if classification == "ambiguous":
            counts["ambiguous_top2_accepted_count"] += 1
        counts["ferrum_oracle_exact_count"] += int(ferrum_argmax == oracle_argmax)
        counts["llama_oracle_exact_count"] += int(llama_argmax == oracle_argmax)
        counts["external_flip_count"] += int(llama_argmax != oracle_argmax)
        decisions.append(
            {
                "index": index,
                "oracle_margin": margin,
                "oracle_top2_token_ids": oracle_top2,
                "oracle_top2_logits": oracle_top2_logits,
                "ferrum_argmax_token_id": ferrum_argmax,
                "llama_argmax_token_id": llama_argmax,
                "classification": classification,
                "status": "pass",
            }
        )
    metrics = {
        label: {
            "minimum_cosine": minimum_cosines[label],
            "maximum_relative_l2": maximum_relative_l2[label],
            "maximum_abs": maximum_abs[label],
        }
        for label in minimum_cosines
    }
    validate_metric_bounds(metrics["ferrum_vs_oracle"], bounds)
    comparison = {
        "schema_version": 1,
        "status": "pass",
        "decision_count": DECISION_COUNT,
        "vocabulary_size": VOCABULARY_SIZE,
        "raw_sets": {
            "ferrum": {
                "decision_count": DECISION_COUNT,
                "row_bytes": ROW_BYTES,
                "ordered_sha256": ordered_raw_sha256(ferrum_paths),
            },
            "llama": {
                "decision_count": DECISION_COUNT,
                "row_bytes": ROW_BYTES,
                "ordered_sha256": ordered_raw_sha256(llama_paths),
            },
            "oracle": {
                "decision_count": DECISION_COUNT,
                "row_bytes": ROW_BYTES,
                "ordered_sha256": ordered_raw_sha256([oracle_path]),
            },
        },
        "metrics": metrics,
        "decisions": decisions,
        "summary": counts,
    }
    require(not out.exists(), f"comparison output already exists: {out}")
    write_json_atomic(out, comparison)
    validated_counts = validate_comparison(
        out,
        ferrum_paths=ferrum_paths,
        llama_paths=llama_paths,
        oracle_path=oracle_path,
        oracle_decisions=oracle_decisions,
        bounds=bounds,
    )
    require(validated_counts == counts, "comparison write/read validation differs")
    return comparison


def validate_manifest(path: Path, tolerance_row: dict[str, Any]) -> dict[str, Any]:
    document = exact_object(read_json(path, "same-history manifest"), ROOT_FIELDS, "same-history manifest")
    require(document["schema_version"] == SCHEMA_VERSION and document["status"] == "pass", "same-history manifest is not PASS")
    require(document["source_dirty"] is False, "same-history source must be clean")
    source_git_sha = git_sha_text(document["source_git_sha"], "source_git_sha")
    source_tree_sha = git_sha_text(document["source_tree_sha"], "source_tree_sha")
    require(
        document["model_id"] == MODEL_ID
        and document["model_revision"] == MODEL_REVISION
        and document["model_file_sha256"] == MODEL_SHA256
        and document["backend"] == "metal",
        "same-history model/backend identity differs",
    )
    require(
        document["tolerance_id"] == TOLERANCE_ID
        and finite(document["oracle_ambiguity_margin"], "oracle_ambiguity_margin") == ORACLE_AMBIGUITY_MARGIN,
        "same-history oracle policy differs",
    )
    require(document["teacher_token_count"] == DECISION_COUNT, "same-history teacher denominator differs")
    require(document["exception_count"] == document["waiver_count"] == 0, "same-history manifest contains waiver/exception")
    prompt_sha = sha256_text(document["prompt_token_ids_sha256"], "prompt_token_ids_sha256")
    teacher_sha = sha256_text(document["teacher_token_ids_sha256"], "teacher_token_ids_sha256")
    sha256_text(document["prompt_sha256"], "prompt_sha256")
    ferrum_dir = Path(document["ferrum_capture_dir"]).expanduser().resolve()
    llama_dir = Path(document["llama_capture_dir"]).expanduser().resolve()
    oracle_dir = Path(document["oracle_dir"]).expanduser().resolve()
    comparison_path = Path(document["comparison"]).expanduser().resolve()
    ferrum_paths, prompt_tokens, teacher_tokens, ferrum_summary = ferrum_paths_and_history(ferrum_dir)
    require(token_ids_sha256(prompt_tokens) == prompt_sha, "same-history prompt token SHA differs from Ferrum capture")
    require(token_ids_sha256(teacher_tokens) == teacher_sha, "same-history teacher token SHA differs from Ferrum capture")
    llama_paths, _llama_manifest = validate_llama_manifest(
        llama_dir,
        prompt_tokens=prompt_tokens,
        teacher_tokens=teacher_tokens,
    )
    oracle_path, oracle_decisions, _oracle_report = validate_oracle_report(
        oracle_dir,
        source_git_sha=source_git_sha,
        prompt_token_sha256=prompt_sha,
        teacher_token_sha256=teacher_sha,
    )
    bounds = validate_tolerance_row(tolerance_row)
    summary = validate_comparison(
        comparison_path,
        ferrum_paths=ferrum_paths,
        llama_paths=llama_paths,
        oracle_path=oracle_path,
        oracle_decisions=oracle_decisions,
        bounds=bounds,
    )
    return {
        "schema_version": 1,
        "status": "pass",
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "prompt_id": document["prompt_id"],
        "prompt_sha256": document["prompt_sha256"],
        "prompt_token_ids_sha256": prompt_sha,
        "teacher_token_ids_sha256": teacher_sha,
        "teacher_token_count": DECISION_COUNT,
        "tolerance_id": TOLERANCE_ID,
        "plan_hash": ferrum_summary["plan_hash"],
        **summary,
    }


def self_test() -> None:
    robust = array("f", [4.0, 3.0, 2.0, 1.0])
    argmax, top2, logits = argmax_and_top2(robust, "robust")
    require(argmax == 0 and top2 == [0, 1] and logits == [4.0, 3.0], "top2 fixture differs")
    require(robust == array("f", [4.0, 3.0, 2.0, 1.0]), "top2 mutated its input")
    bounds = {"cosine_min": 0.999, "relative_l2_max": 0.01, "max_abs_max": 0.1}
    validate_metric_bounds(
        {"minimum_cosine": 0.9999, "maximum_relative_l2": 0.001, "maximum_abs": 0.02},
        bounds,
    )
    rejected = 0
    for metric, marker in (
        ({"minimum_cosine": 0.998, "maximum_relative_l2": 0.001, "maximum_abs": 0.02}, "cosine"),
        ({"minimum_cosine": 1.0, "maximum_relative_l2": 0.02, "maximum_abs": 0.02}, "relative L2"),
        ({"minimum_cosine": 1.0, "maximum_relative_l2": 0.001, "maximum_abs": 0.2}, "max abs"),
    ):
        try:
            validate_metric_bounds(metric, bounds)
        except GateError as error:
            require(marker.lower() in str(error).lower(), f"metric fixture rejected incorrectly: {error}")
            rejected += 1
    require(rejected == 3, "metric rejection denominator differs")
    require(
        decision_classification(ORACLE_AMBIGUITY_MARGIN) == "robust"
        and decision_classification(ORACLE_AMBIGUITY_MARGIN - 1.0e-6) == "ambiguous",
        "oracle ambiguity boundary differs",
    )
    enforce_decision_policy(
        index=0,
        classification="robust",
        ferrum_argmax=7,
        oracle_argmax=7,
        oracle_top2=[7, 8],
    )
    enforce_decision_policy(
        index=1,
        classification="ambiguous",
        ferrum_argmax=8,
        oracle_argmax=7,
        oracle_top2=[7, 8],
    )
    for name, kwargs, marker in (
        (
            "robust runner-up",
            {"index": 2, "classification": "robust", "ferrum_argmax": 8, "oracle_argmax": 7, "oracle_top2": [7, 8]},
            "robust",
        ),
        (
            "ambiguous rank-3",
            {"index": 3, "classification": "ambiguous", "ferrum_argmax": 9, "oracle_argmax": 7, "oracle_top2": [7, 8]},
            "outside oracle top2",
        ),
    ):
        try:
            enforce_decision_policy(**kwargs)
        except GateError as error:
            require(marker in str(error), f"{name} rejected incorrectly: {error}")
        else:
            raise GateError(f"{name} unexpectedly passed")
    with __import__("tempfile").TemporaryDirectory(prefix="g08a-same-history-selftest-") as temporary:
        root = Path(temporary)
        raw = root / "row.f32"
        robust.tofile(raw.open("wb"))
        with raw.open("rb") as handle:
            loaded = load_f32_row(handle, 4, "fixture")
            require(not handle.read(1), "fixture has trailing bytes")
        require(loaded == robust, "f32 fixture round-trip differs")
        raw.write_bytes(raw.read_bytes()[:-1])
        try:
            with raw.open("rb") as handle:
                load_f32_row(handle, 4, "truncated fixture")
        except GateError as error:
            require("truncated" in str(error), "truncated fixture rejected incorrectly")
        else:
            raise GateError("truncated fixture unexpectedly passed")
        nonfinite = root / "nonfinite.f32"
        values = array("f", [1.0, math.nan, 0.0, -1.0])
        with nonfinite.open("wb") as handle:
            values.tofile(handle)
        try:
            with nonfinite.open("rb") as handle:
                load_f32_row(handle, 4, "nonfinite fixture")
        except GateError as error:
            require("NaN or Inf" in str(error), "nonfinite fixture rejected incorrectly")
        else:
            raise GateError("nonfinite fixture unexpectedly passed")
    print(SELFTEST_PASS)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--self-test", action="store_true")
    result.add_argument("--manifest", type=Path)
    result.add_argument("--tolerance-row", type=Path)
    result.add_argument("--build-comparison", action="store_true")
    result.add_argument("--ferrum-capture", type=Path)
    result.add_argument("--llama-capture", type=Path)
    result.add_argument("--oracle-dir", type=Path)
    result.add_argument("--out", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if args.build_comparison:
            required = (
                args.ferrum_capture,
                args.llama_capture,
                args.oracle_dir,
                args.tolerance_row,
                args.out,
            )
            require(all(value is not None for value in required), "comparison inputs and --out are required")
            tolerance_row = read_json(args.tolerance_row, "same-history tolerance row")
            comparison = build_comparison(
                ferrum_capture_dir=args.ferrum_capture,
                llama_capture_dir=args.llama_capture,
                oracle_dir=args.oracle_dir,
                tolerance_row=tolerance_row,
                out=args.out,
            )
            print(json.dumps(comparison["summary"], sort_keys=True))
            return 0
        require(args.manifest is not None and args.tolerance_row is not None, "--manifest and --tolerance-row are required")
        tolerance_row = read_json(args.tolerance_row, "same-history tolerance row")
        summary = validate_manifest(args.manifest, tolerance_row)
        print(f"{PASS_PREFIX}: {args.manifest.expanduser().resolve()}")
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (GateError, OSError, ValueError) as error:
        print(f"{FAIL_PREFIX}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
